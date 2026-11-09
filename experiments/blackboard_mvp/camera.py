"""Recoverable camera geometry from bounded, local static-anchor observations.

An appearance difference is a disturbance cue, not proof of a camera movement.
Bad observations never replace the trusted scene reference or stop future search.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np

from .models import TransformRef


class CameraMonitor:
    def __init__(self, anchor_regions=None, excluded_regions=None, recovery_observations=3):
        if type(recovery_observations) is not int or recovery_observations < 1:
            raise ValueError('recovery_observations must be a positive integer')
        self.anchor_regions = anchor_regions
        self.excluded_regions = excluded_regions or []
        for region in (anchor_regions or []) + self.excluded_regions:
            if len(region) != 4 or any(type(v) is not int for v in region) or min(region[:2]) < 0 or min(region[2:]) <= 0:
                raise ValueError('Camera regions require nonnegative integer XY and positive WH')
        self.recovery_observations = recovery_observations
        self._reference = self._previous = None
        self._mask = self._points = None
        self._anchor_distance = None
        self._last_time = None
        self._frame_ids = set()
        self._verified = False
        self._recovering = False
        self._count = 0
        self._recovery_signatures = set()
        self._last_transform = None
        self._reference_id = None
        self.geometry_epoch = 0
        self.scene_id = 0
        self.history = []

    def _anchor_mask(self, shape):
        height, width = shape
        mask = np.zeros(shape, np.uint8) if self.anchor_regions is not None else np.full(shape,255,np.uint8)
        for x,y,w,h in self.anchor_regions or []:
            if x+w > width or y+h > height:
                raise ValueError('Camera anchor outside source image')
            mask[y:y+h,x:x+w] = 255
        # Explicit, human-selected static anchors can lie inside a broad
        # component *search* region. Search range is not a moving-board mask.
        # Only the default derived-outside mode applies component exclusions.
        if self.anchor_regions is None:
            for x,y,w,h in self.excluded_regions:
                if x+w > width or y+h > height:
                    raise ValueError('Camera exclusion outside source image')
                mask[y:y+h,x:x+w] = 0
        # Keep feature centres away from untrusted board boundaries. This does
        # not certify the whole LK window; final FB/residual/support checks apply.
        return cv2.erode(mask, np.ones((5,5),np.uint8))

    @staticmethod
    def _distance(mask):
        # Explicit zero image border: an anchor touching the image edge does not
        # acquire an infinite distance merely because the domain is all-white.
        padded=cv2.copyMakeBorder(mask,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
        return cv2.distanceTransform(padded,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]

    @staticmethod
    def _features(gray, mask):
        height,width=gray.shape
        # 25x25 LK footprint plus interpolation margin. A narrow anchor cannot
        # borrow contrast from the moving board next to it. LK below uses level0.
        safe=(CameraMonitor._distance(mask)>=14).astype(np.uint8)*255
        points=[]
        for row in range(3):
            for col in range(3):
                x0,x1=width*col//3,width*(col+1)//3
                y0,y1=height*row//3,height*(row+1)//3
                p=cv2.goodFeaturesToTrack(gray[y0:y1,x0:x1],maxCorners=70,qualityLevel=.015,
                                         minDistance=6,mask=safe[y0:y1,x0:x1])
                if p is not None:
                    p=p.reshape(-1,2)+[x0,y0]
                    points.extend(p.tolist())
        return np.array(points,np.float32).reshape(-1,1,2)

    @staticmethod
    def _descriptor_footprints(keypoints, descriptors, distance):
        if descriptors is None:
            return [],None,0
        keep=[]
        h,w=distance.shape
        for i,kp in enumerate(keypoints):
            x,y=int(round(kp.pt[0])),int(round(kp.pt[1]))
            # Bound the rotated patch square and bilinear interpolation support,
            # including the actual pyramid-scaled keypoint size.
            radius=float(kp.size)*(.5*np.sqrt(2))+2
            if 0<=x<w and 0<=y<h and distance[y,x]>=radius:
                keep.append(i)
        return [keypoints[i] for i in keep], descriptors[keep] if keep else None, len(keypoints)-len(keep)

    @staticmethod
    def _coverage(points, shape):
        if len(points)<2:
            return {'support_regions':[], 'spatial_span':[0.,0.], 'support_count':0}
        h,w=shape; points=np.asarray(points).reshape(-1,2)
        span=np.ptp(points,axis=0)/[w,h]
        cells=sorted({(min(2,int(p[1]*3/h)),min(2,int(p[0]*3/w))) for p in points})
        return {'support_regions':[list(c) for c in cells], 'spatial_span':span.tolist(), 'support_count':len(cells)}

    @classmethod
    def _distributed(cls, points, shape):
        c=cls._coverage(points,shape)
        return len(points)>=12 and c['support_count']>=3 and c['spatial_span'][0]>=.20 and c['spatial_span'][1]>=.10

    def _fit(self, starts, ends, shape, method, fb_error=None):
        proof={'method':method,'matches':int(len(starts)),'inliers':0,'residual_px':None,
               'forward_backward_error_px':fb_error,'model':None}
        if len(starts)<12:
            return None,proof
        starts=np.asarray(starts,np.float32).reshape(-1,2); ends=np.asarray(ends,np.float32).reshape(-1,2)
        # First prefer a translation. Multiple moving regions cannot be swallowed
        # by an unconstrained homography.
        shift=ends-starts; centre=np.median(shift,axis=0)
        error=np.linalg.norm(shift-centre,axis=1); inliers=error<=1.5
        model='translation'; matrix=np.eye(3,dtype=float)
        matrix[:2,2]=-centre
        if inliers.sum()<12 or inliers.mean()<.65:
            affine,inlier_mask=cv2.estimateAffinePartial2D(ends,starts,method=cv2.RANSAC,ransacReprojThreshold=1.5,
                                                         maxIters=500,confidence=.995,refineIters=10)
            if affine is None:
                return None,proof
            matrix[:2]=affine
            scale=float(np.sqrt(affine[0,0]**2+affine[1,0]**2))
            angle=float(abs(np.degrees(np.arctan2(affine[1,0],affine[0,0]))))
            if not .90<=scale<=1.10 or angle>8:
                proof['rejected_model']='unbounded_similarity'; return None,proof
            predicted=cv2.transform(ends.reshape(-1,1,2),affine).reshape(-1,2)
            error=np.linalg.norm(predicted-starts,axis=1)
            inliers=error<=1.5; model='similarity'
        proof.update(inliers=int(inliers.sum()),inlier_ratio=float(inliers.mean()),model=model)
        proof.update(self._coverage(starts[inliers],shape))
        if not self._distributed(starts[inliers],shape) or inliers.mean()<.65:
            proof['rejection']='insufficient_distributed_consensus';return None,proof
        if model=='translation':
            centre=np.median(shift[inliers],axis=0); matrix[:2,2]=-centre
            error=np.linalg.norm(shift-centre,axis=1)
        residual=float(np.median(error[inliers])); proof['residual_px']=residual
        if not np.isfinite(matrix).all() or residual>1.0:
            proof['rejection']='geometry_residual';return None,proof
        proof['translation_to_scene_px']=matrix[:2,2].tolist()
        return matrix,proof

    def _estimate(self, gray):
        ref_norm=cv2.equalizeHist(self._reference); cur_norm=cv2.equalizeHist(gray)
        p=self._points
        attempts=[]
        if len(p)>=12:
            q,status,_=cv2.calcOpticalFlowPyrLK(ref_norm,cur_norm,p,None,winSize=(25,25),maxLevel=0,
                                             criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
            if q is not None:
                back,back_status,_=cv2.calcOpticalFlowPyrLK(cur_norm,ref_norm,q,None,winSize=(25,25),maxLevel=0)
                if back is not None:
                    s=p.reshape(-1,2); e=q.reshape(-1,2)
                    fb=np.linalg.norm(back.reshape(-1,2)-s,axis=1)
                    h,w=gray.shape
                    valid=(status.ravel()>0)&(back_status.ravel()>0)&(fb<1.2)&np.isfinite(e).all(axis=1)
                    valid&=(e[:,0]>=0)&(e[:,0]<w)&(e[:,1]>=0)&(e[:,1]<h)
                    safe_x=np.rint(np.nan_to_num(e[:,0],nan=-1,posinf=-1,neginf=-1)).clip(0,w-1).astype(int)
                    safe_y=np.rint(np.nan_to_num(e[:,1],nan=-1,posinf=-1,neginf=-1)).clip(0,h-1).astype(int)
                    valid&=self._anchor_distance[safe_y,safe_x]>=14
                    matrix,proof=self._fit(s[valid],e[valid],gray.shape,'static_anchor_lk',float(np.median(fb[valid])) if valid.any() else None)
                    proof.update(lk_max_level=0,lk_window_size=25,reference_footprint_points=int(len(p)),target_footprint_points=int(valid.sum()),anchor_support_window_verified=matrix is not None)
                    attempts.append(proof)
                    if matrix is not None:
                        proof['attempts']=[dict(item) for item in attempts]; return matrix,proof
        # Conservative descriptor recapture. Descriptor centres AND complete
        # footprints must stay in the trusted domain. No dilation into a broad
        # board search region can be treated as static camera support.
        orb=cv2.ORB_create(nfeatures=1800,edgeThreshold=7,patchSize=15,fastThreshold=7)
        a,da=orb.detectAndCompute(self._reference,self._mask)
        b,db=orb.detectAndCompute(gray,self._mask)
        a,da,rejected_ref=self._descriptor_footprints(a,da,self._anchor_distance)
        b,db,rejected_target=self._descriptor_footprints(b,db,self._anchor_distance)
        if da is not None and db is not None and len(db)>=2:
            pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
            matches=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance<.70*n.distance and m.distance<=40]
            unique={}
            for m in sorted(matches,key=lambda m:m.distance): unique.setdefault(m.trainIdx,m)
            matches=list(unique.values())
            starts=np.array([a[m.queryIdx].pt for m in matches],np.float32).reshape(-1,2)
            ends=np.array([b[m.trainIdx].pt for m in matches],np.float32).reshape(-1,2)
            matrix,proof=self._fit(starts,ends,gray.shape,'static_anchor_orb')
            proof.update(reference_footprint_features=len(a),target_footprint_features=len(b),rejected_reference_footprints=rejected_ref,rejected_target_footprints=rejected_target,anchor_support_window_verified=matrix is not None,descriptor_footprint_rule='sqrt(2)*keypoint_size/2+2px')
            attempts.append(proof)
            if matrix is not None:
                proof['attempts']=[dict(item) for item in attempts];return matrix,proof
        return None,{'method':'bounded_anchor_search','matches':max([p.get('matches',0) for p in attempts],default=0),
                     'inliers':max([p.get('inliers',0) for p in attempts],default=0),'attempts':attempts,
                     'anchor_support_window_verified':False,'rejection':'no_reliable_distributed_geometry',
                     'reference_footprint_points':int(len(p)),'orb_reference_footprint_features':len(a),'orb_target_footprint_features':len(b),
                     'rejected_reference_footprints':rejected_ref,'rejected_target_footprints':rejected_target}

    def observe(self, frame, frame_id, timestamp_ms, force_reconfirm=False):
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:
            raise ValueError('Expected uint8 BGR source frame')
        if type(timestamp_ms) is not int or timestamp_ms<0 or (self._last_time is not None and timestamp_ms<=self._last_time):
            raise ValueError('Camera observation timestamps must increase')
        if not isinstance(frame_id,str) or not frame_id or frame_id in self._frame_ids:
            raise ValueError('Camera frame IDs must be unique/nonempty')
        self._last_time=timestamp_ms;self._frame_ids.add(frame_id)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        matrix=None; reasons=[]; disturbance='none'; proof={}; recovered=False
        if self._reference is None:
            self._mask=self._anchor_mask(gray.shape);self._anchor_distance=self._distance(self._mask);self._reference=gray.copy();self._reference_id=frame_id
            self._points=self._features(gray,self._mask)
            proof={'method':'manual_scene_origin','matches':int(len(self._points)),'inliers':int(len(self._points)),
                   'residual_px':0., **self._coverage(self._points,gray.shape),
                   'anchor_source':'explicit_static_regions' if self.anchor_regions is not None else 'outside_configured_component_search_regions',
                   'reference_footprint_points':int(len(self._points)),'lk_max_level':0,'lk_window_size':25,
                   'anchor_support_window_verified':self._distributed(self._points,gray.shape)}
            if self._distributed(self._points,gray.shape) and not force_reconfirm:
                self._verified=True;self._count=self.recovery_observations;matrix=np.eye(3)
                status='verified';reasons=['manual_scene_origin_with_distributed_anchor_support']
            else:
                self._recovering=True;status='unobservable';reasons=['insufficient_static_anchor_support']
        elif gray.shape!=self._reference.shape:
            self._verified=False;self._recovering=True;self._count=0;self._recovery_signatures.clear()
            status='manual_review_required';disturbance='signal_failure';reasons=['source_dimensions_changed']
        else:
            matrix,proof=self._estimate(gray)
            if self._previous is not None and self._previous.shape==gray.shape:
                changed=np.abs(gray.astype(np.int16)-self._previous.astype(np.int16))>=25
                valid=self._mask>0
                proof['anchor_changed_fraction']=float(changed[valid].mean()) if valid.any() else None
                proof['anchor_brightness_delta']=float(np.median(gray[valid].astype(float)-self._previous[valid].astype(float))) if valid.any() else None
            if force_reconfirm:
                self._verified=False;self._recovering=True;self._count=0;self._recovery_signatures.clear()
                matrix=None;status='degraded';disturbance='appearance_disturbance_unknown';reasons=['manual_reconfirmation_requested']
            elif matrix is None:
                had=self._verified
                self._verified=False;self._recovering=True;self._count=0;self._recovery_signatures.clear()
                status='degraded' if had else 'unobservable'
                disturbance='appearance_disturbance_unknown'
                if proof.get('anchor_changed_fraction',0) is not None and proof.get('anchor_changed_fraction',0)>.75:
                    disturbance='scene_cut_candidate'
                if float(gray.std()) < .2:
                    disturbance='signal_failure'
                reasons=['insufficient_geometric_evidence','old_scene_reference_preserved','future_search_enabled']
            else:
                delta=float(np.linalg.norm(matrix[:2,2]))
                if delta>1.5 or abs(matrix[0,0]-1)>.005 or abs(matrix[1,0])>.005:
                    disturbance='camera_motion_supported'
                elif abs(proof.get('anchor_brightness_delta') or 0)>8:
                    disturbance='photometric_change'
                elif (proof.get('anchor_changed_fraction') or 0)>.10:
                    disturbance='local_dynamic'
                if self._recovering:
                    signature=hashlib.blake2b(gray[self._mask>0].tobytes(),digest_size=12).digest()
                    distinct=signature not in self._recovery_signatures
                    proof['recovery_distinct_anchor_pixels']=distinct
                    if distinct:
                        self._recovery_signatures.add(signature); self._count+=1
                    if self._count<self.recovery_observations:
                        status='reinitializing';reasons=['collecting_independent_geometric_recovery_evidence']
                        if not distinct: reasons.append('duplicate_anchor_image_not_new_recovery_evidence')
                        matrix=None
                    else:
                        self._verified=True;self._recovering=False;self.geometry_epoch+=1
                        recovered=True;status='verified';reasons=['old_scene_anchor_mapping_reacquired','board_identity_requires_independent_confirmation']
                else:
                    self._verified=True;status='verified';self._count=self.recovery_observations
                    reasons=['distributed_static_anchor_geometry_verified']
        transform_ref = None
        if matrix is not None:
            transform_ref = TransformRef(f'camera:{self.geometry_epoch}:{frame_id}', 'source_screen', f'scene:{self.scene_id}', matrix.tolist(), self.geometry_epoch, error_px=proof.get('residual_px')).to_dict()
            self._last_transform={'frame_id':frame_id,'timestamp_ms':timestamp_ms,'matrix':matrix.tolist(),'geometry_epoch':self.geometry_epoch}
        result={'frame_id':frame_id,'timestamp_ms':timestamp_ms,'reliability':status,'disturbance_type':disturbance,
                'measured_transform_to_scene':matrix.tolist() if matrix is not None else None,
                'transform_ref':transform_ref,
                'last_valid_transform':self._last_transform,'geometry_epoch':self.geometry_epoch,'scene_id':self.scene_id,
                'evidence':proof,'reason_codes':reasons,'search_attempted':True,'recovery_count':self._count,
                'reference_frame_id':self._reference_id,'recovered':recovered,
                'identity_inheritance_allowed':False,'comparison_requires_independent_board_identity':True}
        self.history.append(result);self._previous=gray.copy()
        return result
