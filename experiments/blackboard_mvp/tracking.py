"""Local physical-board hypotheses, measurements and identity association.

This module generates no content events.  Initial board-local coordinates are
immutable; a failed search never returns the previous ROI as a measurement.
Mechanical linkage produces labelled predictions and is never identity evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math

import cv2
import numpy as np

from .geometry import intersect as _intersect


def _project(points, transform):
    points = np.asarray(points, dtype=np.float64)
    hom = np.column_stack((points, np.ones(len(points)))) @ transform.T
    return hom[:, :2] / hom[:, 2:3]


def _screen_roi(scene_origin, size, scene_to_screen):
    x, y = scene_origin
    w, h = size
    corners = _project([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], scene_to_screen)
    lower = np.floor(corners.min(axis=0)+.1).astype(int)
    upper = np.ceil(corners.max(axis=0)-.1).astype(int)
    return [int(lower[0]),int(lower[1]),max(1,int(upper[0]-lower[0])),max(1,int(upper[1]-lower[1]))]


@dataclass(frozen=True)
class TrackingConfig:
    minimum_inliers: int = 8
    translation_tolerance_px: float = 2.0
    descriptor_ratio: float = .72
    maximum_hypotheses: int = 3
    target_features_per_tile: int = 350
    maximum_gallery_entries: int = 1
    minimum_appearance_fraction: float = .65

    def __post_init__(self):
        if self.minimum_inliers < 4 or self.maximum_hypotheses < 1 or self.target_features_per_tile < 20:
            raise ValueError('Invalid bounded tracking configuration')
        if not 0 < self.descriptor_ratio < 1 or not 0 < self.minimum_appearance_fraction < 1 or not math.isfinite(self.translation_tolerance_px) or self.translation_tolerance_px <= 0:
            raise ValueError('Invalid tracking thresholds')


class DynamicBoardTracker:
    """Automatic tracking with separate geometry and physical identity states.

    Search descriptors are allocated by tile, not one full-viewport budget.  Each
    board owns an immutable initialization reference.  Coherent translation groups
    compete independently; a majority of *all* matches is not required.
    """
    def __init__(self, boards, assemblies, config=None):
        self.boards = list(boards)
        self.assemblies = {a.assembly_id:a for a in assemblies}
        self.config = config or TrackingConfig()
        if not self.boards or len({b.board_id for b in boards}) != len(boards):
            raise ValueError('Physical board IDs must be unique')
        if any(b.assembly_id not in self.assemblies for b in boards):
            raise ValueError('Each physical board needs an assembly search region')
        for aid in self.assemblies:
            members = [b for b in self.boards if b.assembly_id == aid]
            if len(members) != 2 or {b.layer for b in members} != {'outer','inner'}:
                raise ValueError('Each assembly needs one outer and one inner physical board')
        self._references = {}
        self._states = {}
        self._pair_anchors = {}
        self._last_time = None
        self._frame_ids = set()
        self.history = []
        self._sift = cv2.SIFT_create(nfeatures=1200, contrastThreshold=.012, edgeThreshold=15)
        self._target_sift = cv2.SIFT_create(nfeatures=self.config.target_features_per_tile,
                                           contrastThreshold=.012, edgeThreshold=15)

    def _initialize(self, frame):
        height,width = frame.shape[:2]
        for board in self.boards:
            x,y,w,h = board.roi
            if min(x,y) < 0 or x+w > width or y+h > height:
                raise ValueError('Initial ROI lies outside source frame')
            crop = frame[y:y+h,x:x+w].copy()
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            mask = np.full(gray.shape,255,np.uint8)
            # Initial overlap contains the front board, not the rear board's
            # identity.  Only the actually exposed rear surface may be a template.
            if board.layer == 'inner':
                outer = next(b for b in self.boards if b.assembly_id==board.assembly_id and b.layer=='outer')
                overlap = _intersect(board.roi,outer.roi)
                if overlap:
                    ox,oy,ow,oh=overlap;mask[oy-y:oy-y+oh,ox-x:ox-x+ow]=0
            mask = cv2.erode(mask,np.ones((5,5),np.uint8))
            kps,des = self._sift.detectAndCompute(gray,mask)
            self._references[board.board_id] = {'gray':gray,'mask':mask,
                'points':np.array([k.pt for k in kps],dtype=np.float32).reshape(-1,2),
                'descriptors':des,'source':'manual_initial_visible_surface','source_frame_id':None}

    def _target(self, gray, assembly):
        vx,vy,vw,vh = assembly.viewport
        clip = _intersect([vx,vy,vw,vh],[0,0,gray.shape[1],gray.shape[0]])
        if clip is None:
            return np.empty((0,2),np.float32),None
        vx,vy,vw,vh=clip
        points,desc=[],[]
        # Three vertical bands prevent a feature-rich outer board starving a
        # faint lower board, with two columns to enforce lateral coverage.
        for row in range(3):
            for col in range(2):
                x0=vx+vw*col//2;x1=vx+vw*(col+1)//2
                y0=vy+vh*row//3;y1=vy+vh*(row+1)//3
                patch=gray[y0:y1,x0:x1]
                kp,ds=self._target_sift.detectAndCompute(patch,None)
                if ds is not None:
                    points.extend((k.pt[0]+x0,k.pt[1]+y0) for k in kp)
                    desc.append(ds)
        return np.array(points,np.float32).reshape(-1,2), np.concatenate(desc) if desc else None

    def _appearance(self, frame, reference, origin, scene_to_screen, outer_roi=None):
        h,w=reference['gray'].shape
        # Canonical board-local output; source pixels outside frame remain masked.
        to_screen=scene_to_screen @ np.array([[1,0,origin[0]],[0,1,origin[1]],[0,0,1]],float)
        try: to_local=np.linalg.inv(to_screen)
        except np.linalg.LinAlgError: return None
        current=cv2.warpPerspective(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),to_local,(w,h))
        valid=cv2.warpPerspective(np.full(frame.shape[:2],255,np.uint8),to_local,(w,h),flags=cv2.INTER_NEAREST)>0
        common=valid & (reference['mask']>0)
        if outer_roi is not None:
            sx,sy,sw,sh=outer_roi
            cover=np.zeros(frame.shape[:2],np.uint8)
            c=_intersect(outer_roi,[0,0,frame.shape[1],frame.shape[0]])
            if c:
                x,y,cw,ch=c;cover[y:y+ch,x:x+cw]=255
                common &= cv2.warpPerspective(cover,to_local,(w,h),flags=cv2.INTER_NEAREST)==0
        if np.count_nonzero(common)<max(100,int(w*h*.035)):
            return None
        before=reference['gray'].astype(np.float32);after=current.astype(np.float32)
        offset=float(np.median((after-before)[common]))
        error=np.abs(after-before-offset)
        return {'known_visible_pixels':int(np.count_nonzero(common)),
                'appearance_fraction':float(np.mean(error[common]<=30)),
                'median_appearance_error_px':float(np.median(error[common])),
                'photometric_offset':offset}

    def _hypotheses(self, frame, board, target, camera_transform, outer_roi):
        ref=self._references[board.board_id]
        target_points,target_des=target
        des=ref['descriptors']
        if des is None or target_des is None or len(des)<self.config.minimum_inliers or len(target_des)<2:
            return [], {'reason':'insufficient_distinctive_features','reference_features':len(ref['points']),
                        'target_features':len(target_points),'matches':0}
        pairs=cv2.BFMatcher(cv2.NORM_L2).knnMatch(des,target_des,k=2)
        matches=[m for p in pairs if len(p)==2 for m,n in [p]
                 if m.distance < self.config.descriptor_ratio*n.distance]
        # One target feature cannot support several physical-source features.
        unique={}
        for m in sorted(matches,key=lambda m:m.distance): unique.setdefault(m.trainIdx,m)
        matches=list(unique.values())
        proof={'reference_features':len(ref['points']),'target_features':len(target_points),'matches':len(matches)}
        if len(matches)<self.config.minimum_inliers: return [],{**proof,'reason':'insufficient_unique_matches'}
        starts=ref['points'][[m.queryIdx for m in matches]].astype(float)
        ends=target_points[[m.trainIdx for m in matches]].astype(float)
        scene_ends=_project(ends,camera_transform)
        offsets=scene_ends-starts
        remaining=np.ones(len(matches),bool);hypotheses=[]
        inv=np.linalg.inv(camera_transform)
        # Bounded displacement-mode clustering.  Different moving surfaces do
        # not dilute one another's inlier fraction.
        for _ in range(self.config.maximum_hypotheses):
            indices=np.flatnonzero(remaining)
            if len(indices)<self.config.minimum_inliers: break
            distances=np.linalg.norm(offsets[indices,None,:]-offsets[None,indices,:],axis=2)
            seed=indices[np.argmax(np.sum(distances<=self.config.translation_tolerance_px,axis=1))]
            group=remaining & (np.linalg.norm(offsets-offsets[seed],axis=1)<=self.config.translation_tolerance_px)
            origin=np.median(offsets[group],axis=0)
            group=remaining & (np.linalg.norm(offsets-origin,axis=1)<=self.config.translation_tolerance_px)
            remaining[group]=False
            if np.count_nonzero(group)<self.config.minimum_inliers: continue
            origin=np.median(offsets[group],axis=0)
            spread=np.ptp(starts[group],axis=0)
            w,h=board.roi[2:]
            if spread[0]<min(40,w*.18) or spread[1]<min(20,h*.12): continue
            # Rails and screen overlays can span the whole ROI while the panel
            # underneath moves. Edge-only support cannot confirm its identity.
            # Require four distinct interior locations, not SIFT orientations
            # repeated at the same location. This remains an evidence gate,
            # not proof that every interior texture belongs to the panel.
            support_points = starts[group]
            interior = support_points[
                (support_points[:, 0] >= w * .15) & (support_points[:, 0] <= w * .85) &
                (support_points[:, 1] >= h * .15) & (support_points[:, 1] <= h * .85)
            ]
            interior_count = len(np.unique(np.rint(interior / 3).astype(int), axis=0))
            if interior_count < 4:
                proof['reason'] = 'insufficient_interior_identity_support'
                proof.setdefault('rejected_identity_support', []).append({
                    'inliers': int(np.count_nonzero(group)),
                    'unique_interior_support': interior_count,
                    'required_interior_support': 4,
                    'interior_inset_fraction': .15,
                })
                continue
            # A rail-direction prior allows small horizontal localization error,
            # but cannot confirm identity.  Camera motion is already removed.
            if abs(origin[0]-board.roi[0])>max(12,w*.025): continue
            roi=_screen_roi(origin,(w,h),inv)
            if _intersect(roi,self.assemblies[board.assembly_id].viewport) is None: continue
            appearance=self._appearance(frame,ref,origin,inv,outer_roi)
            if appearance is None or appearance['appearance_fraction']<self.config.minimum_appearance_fraction: continue
            residual=float(np.median(np.linalg.norm(offsets[group]-origin,axis=1)))
            support=int(np.count_nonzero(group))
            # Evidence score is for ranking, never an identity probability.
            score=support*appearance['appearance_fraction']/(1+residual)
            hypotheses.append({'screen_roi':roi,'scene_origin':[float(x) for x in origin],
                'inliers':support,'matches':len(matches),'support_spread_px':[float(x) for x in spread],
                'unique_interior_support':interior_count,'required_interior_support':4,'interior_inset_fraction':.15,
                'translation_residual_px':residual,'appearance':appearance,'ranking_score':float(score),
                'target_points':ends[group].tolist(), 'reference_frame_id':ref['source_frame_id']})
        hypotheses.sort(key=lambda p:-p['ranking_score'])
        proof.setdefault('reason','no_acceptable_translation_hypothesis')
        return hypotheses,proof

    def observe(self, frame, frame_id, timestamp_ms, camera, manual_positions=None, oracle=False):
        if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:
            raise ValueError('Expected uint8 BGR frame')
        if type(timestamp_ms) is not int or timestamp_ms<0 or (self._last_time is not None and timestamp_ms<=self._last_time):
            raise ValueError('Observation timestamps must increase strictly')
        if frame_id in self._frame_ids: raise ValueError('Frame IDs must be unique')
        manual_positions=manual_positions or {}
        if not isinstance(manual_positions,dict) or set(manual_positions)-{b.board_id for b in self.boards}:
            raise ValueError('Manual positions must name configured physical boards')
        if self._last_time is not None and manual_positions and not oracle:
            raise ValueError('Future manual positions are ground truth; use explicit oracle mode')
        for bid,roi in manual_positions.items():
            b=next(b for b in self.boards if b.board_id==bid)
            if not isinstance(roi,(list,tuple)) or len(roi)!=4 or any(type(v) is not int for v in roi) or min(roi[2:])<=0 or list(roi[2:])!=list(b.roi[2:]):
                raise ValueError('Manual ROI must preserve physical size and integer coordinates')
        first=self._last_time is None
        if first:
            self._initialize(frame)
            for ref in self._references.values(): ref['source_frame_id']=frame_id
        raw=camera.get('measured_transform_to_scene')
        matrix=np.eye(3) if raw is None else np.asarray(raw,dtype=float)
        if matrix.shape!=(3,3) or not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix))<1e-10:
            raise ValueError('Camera transform must be finite invertible 3x3')
        camera_verified=camera.get('reliability')=='verified' and raw is not None
        if not camera_verified:
            # Identity here means screen-coordinate search, NOT an assertion
            # that the unobservable camera stayed stationary.
            matrix=np.eye(3)
        geometry_epoch=camera.get('geometry_epoch',0)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        targets={} if first else {aid:self._target(gray,a) for aid,a in self.assemblies.items()}
        proposals={};temporary={};predictions={}
        for b in sorted(self.boards,key=lambda b:b.layer!='outer'):
            old=self._states.get(b.board_id)
            if first or b.board_id in manual_positions:
                roi=list(manual_positions.get(b.board_id,b.roi))
                origin=_project([[roi[0],roi[1]]],matrix)[0]
                proposals[b.board_id]=[{'screen_roi':roi,'scene_origin':origin.tolist(),'inliers':None,
                    'matches':None,'support_spread_px':None,'translation_residual_px':None,'appearance':None,
                    'ranking_score':None,'target_points':[],'reference_frame_id':frame_id}]
                proof={'reason':'manual_initial_identity' if first else 'oracle_manual_checkpoint'}
                certain=True
            else:
                outer=next(x for x in self.boards if x.assembly_id==b.assembly_id and x.layer=='outer')
                outer_roi=temporary.get(outer.board_id,{}).get('measured_screen_roi') if b.layer=='inner' else None
                hypotheses,proof=self._hypotheses(frame,b,targets[b.assembly_id],matrix,outer_roi)
                proposals[b.board_id]=hypotheses
                # Distinctive visual association is independent of whether a
                # screen-to-scene camera mapping is currently observable.
                certain=bool(hypotheses)
                if len(hypotheses)>1 and hypotheses[1]['ranking_score']>=hypotheses[0]['ranking_score']*.80:
                    certain=False;proof['reason']='competing_translation_hypotheses'
                elif hypotheses: proof['reason']='direct_distinctive_visual_support' if camera_verified else 'direct_visual_identity_camera_geometry_unverified'
            for hypothesis in proposals[b.board_id]:
                hypothesis['origin_coordinate_space']='scene' if camera_verified else 'screen_unverified_camera'
                if not camera_verified:
                    hypothesis['unverified_screen_origin']=hypothesis.get('scene_origin')
                    hypothesis['scene_origin']=None
            chosen=proposals[b.board_id][0] if proposals[b.board_id] else None
            measured=list(chosen['screen_roi']) if chosen else None
            previous_origin=old.get('_scene_origin') if old else None
            dt=(timestamp_ms-old['timestamp_ms'])/1000 if old else 0
            last_scene_time=old.get('last_verified_scene_timestamp_ms') if old else None
            prediction_dt=(timestamp_ms-last_scene_time)/1000 if last_scene_time is not None else 0
            velocity=old.get('_velocity',[0.,0.]) if old else [0.,0.]
            predicted_scene=(np.array(previous_origin)+np.array(velocity)*prediction_dt) if previous_origin is not None else np.array(b.roi[:2],float)
            scene_mapping_compatible=bool(camera_verified and old and old.get('last_verified_scene_geometry_epoch')==geometry_epoch and previous_origin is not None)
            if scene_mapping_compatible:
                predicted=_screen_roi(predicted_scene,b.roi[2:],np.linalg.inv(matrix))
                prediction_source='scene_constant_velocity_prediction'
            else:
                predicted=list(old['last_verified_screen_roi']) if old and old.get('last_verified_screen_roi') else list(b.roi)
                prediction_source='last_verified_screen_without_current_scene_mapping'
            if first: predicted=list(measured)
            predictions[b.board_id]=predicted
            identity_recovered=bool(old and certain and old['identity_status']!='confirmed')
            geometry_recovered=bool(old and certain and camera_verified and (old.get('scene_geometry_status')!='verified' or old['geometry_epoch']!=geometry_epoch))
            recovered=identity_recovered or geometry_recovered
            epoch=(old['comparison_epoch'] if old else 0)+(1 if recovered else 0)
            identity_epoch=(old['identity_epoch'] if old else 0)+(1 if identity_recovered else 0)
            moved=False; displacement=None
            scene_observed=bool(certain and camera_verified and chosen and chosen['scene_origin'] is not None)
            continuous_scene_observation=bool(scene_observed and old and old.get('scene_geometry_status')=='verified' and old['geometry_epoch']==geometry_epoch and old['identity_status']=='confirmed' and not recovered)
            if continuous_scene_observation and previous_origin is not None:
                delta=np.array(chosen['scene_origin'])-np.array(previous_origin)
                displacement=[float(v) for v in delta]
                moved=bool(np.linalg.norm(delta)>.75)
            new_velocity=[0.,0.] if scene_observed and recovered else list(velocity)
            if displacement is not None and dt>0:
                new_velocity=(np.array(displacement)/dt).clip(-600,600).tolist()
            temp={'board_id':b.board_id,'assembly_id':b.assembly_id,'layer':b.layer,'label':b.label,
                'frame_id':frame_id,'timestamp_ms':timestamp_ms,'measured_screen_roi':measured,
                'measurement_status':'observed' if measured is not None else 'predicted_only',
                'predicted_screen_roi':predicted,'last_verified_screen_roi':list(measured) if certain else (old['last_verified_screen_roi'] if old else None),
                'last_verified_frame_id':frame_id if certain else (old['last_verified_frame_id'] if old else None),
                'last_verified_timestamp_ms':timestamp_ms if certain else (old['last_verified_timestamp_ms'] if old else None),
                'identity_status':'confirmed' if certain else 'identity_uncertain',
                'visual_association_status':('manual_initialization' if first else ('oracle_manual_not_automatic' if b.board_id in manual_positions else ('confirmed' if certain else 'uncertain'))),
                'identity_evidence':{'source':'manual_initial' if first else ('oracle_manual' if b.board_id in manual_positions else 'visual_feature_translation'),
                    'mechanical_prior_is_identity_evidence':False,'metrics_are_probabilities':False},
                'identity_confidence':{'interpretation':'evidence_metrics_not_probability',
                    'inliers':chosen.get('inliers') if chosen else None,
                    'spatial_support_px':chosen.get('support_spread_px') if chosen else None,
                    'best_second_margin':(chosen['ranking_score']-proposals[b.board_id][1]['ranking_score']) if chosen and len(proposals[b.board_id])>1 else None},
                'motion_status':('moving' if moved else 'stationary') if continuous_scene_observation or (first and scene_observed) else 'unresolved',
                'scene_geometry_status':'verified' if scene_observed else 'unverified',
                'last_verified_scene_timestamp_ms':timestamp_ms if scene_observed else last_scene_time,
                'last_verified_scene_geometry_epoch':geometry_epoch if scene_observed else (old.get('last_verified_scene_geometry_epoch') if old else None),
                'delta_scene_px':displacement,'tracking_source':proof['reason'],'tracking_evidence':proof,
                'search_attempted':not first and b.board_id not in manual_positions,
                'hypotheses':proposals[b.board_id], 'geometry_epoch':geometry_epoch,'scene_id':camera.get('scene_id','scene-0'),
                'identity_epoch':identity_epoch,'comparison_epoch':epoch,'baseline_reset_required':first or recovered,
                'identity_reacquired':identity_recovered,'geometry_reacquired':geometry_recovered,
                'prediction_source':prediction_source,
                'prediction_uncertainty_px':(2. if scene_observed else (old.get('prediction_uncertainty_px',2.) if old else 2.)+max(1.,dt*8)),
                'comparison_eligibility':{'eligible':False,'reasons':['s0_s2_content_disabled']},
                '_scene_origin':chosen['scene_origin'] if scene_observed else previous_origin,
                '_velocity':new_velocity}
            temporary[b.board_id]=temp
        # A target feature cannot establish two different physical identities.
        # Conflict invalidates associations, rather than picking a winner by ID.
        for aid in self.assemblies:
            members=[b for b in self.boards if b.assembly_id==aid]
            a,b=(temporary[x.board_id] for x in members)
            if a['hypotheses'] and b['hypotheses']:
                pa=np.array(a['hypotheses'][0]['target_points']);pb=np.array(b['hypotheses'][0]['target_points'])
                if len(pa) and len(pb):
                    overlap=np.sum(np.min(np.linalg.norm(pa[:,None,:]-pb[None,:,:],axis=2),axis=1)<1.5)
                    if overlap>=max(4,min(len(pa),len(pb))*.4):
                        for state in (a,b):
                            state['identity_status']='identity_uncertain';state['motion_status']='unresolved'
                            state['visual_association_status']='uncertain'
                            state['scene_geometry_status']='unverified'
                            state['tracking_evidence']['reason']='cross_board_feature_competition'
                            state['tracking_source']='cross_board_feature_competition'
                            old=self._states.get(state['board_id'])
                            state['last_verified_screen_roi']=old['last_verified_screen_roi'] if old else None
                            state['last_verified_frame_id']=old['last_verified_frame_id'] if old else None
                            state['last_verified_timestamp_ms']=old['last_verified_timestamp_ms'] if old else None
                            state['_scene_origin']=old['_scene_origin'] if old else None
                            state['_velocity']=list(old['_velocity']) if old else [0.,0.]
                            state['last_verified_scene_timestamp_ms']=old.get('last_verified_scene_timestamp_ms') if old else None
                            state['last_verified_scene_geometry_epoch']=old.get('last_verified_scene_geometry_epoch') if old else None
                            state['measured_screen_roi']=None
                            state['measurement_status']='predicted_only'
                            state['delta_scene_px']=None
                            state['prediction_uncertainty_px']=(old.get('prediction_uncertainty_px',2.) if old else 2.)+8
                            state['comparison_epoch']=old['comparison_epoch'] if old else 0
                            state['identity_epoch']=old['identity_epoch'] if old else 0
                            state['baseline_reset_required']=False;state['identity_reacquired']=False;state['geometry_reacquired']=False
            outer=next(temporary[x.board_id] for x in members if x.layer=='outer')
            inner=next(temporary[x.board_id] for x in members if x.layer=='inner')
            anchor=self._pair_anchors.get(aid)
            if (camera_verified and inner['identity_status']!='confirmed' and outer['identity_status']=='confirmed'
                    and outer['scene_geometry_status']=='verified' and anchor is not None and anchor.get('geometry_epoch')==geometry_epoch):
                delta_y=outer['_scene_origin'][1]-anchor['outer_origin'][1]
                scene=[anchor['inner_origin'][0],anchor['inner_origin'][1]-delta_y]
                inner['predicted_screen_roi']=_screen_roi(scene,next(x.roi[2:] for x in members if x.layer=='inner'),np.linalg.inv(matrix))
                inner['prediction_source']='opposed_motion_soft_prior'
            pair_measured=bool(camera_verified and outer['scene_geometry_status']=='verified' and inner['scene_geometry_status']=='verified')
            if pair_measured:
                self._pair_anchors[aid]={'outer_origin':outer['_scene_origin'],'inner_origin':inner['_scene_origin'],'frame_id':frame_id,'geometry_epoch':geometry_epoch}
            for state in (outer,inner):
                state.setdefault('prediction_source','constant_velocity_or_last_verified')
                state['mechanical_prior']={'identity_evidence':False,'k':1.0,'k_status':'unverified_prior',
                    'source':'paired_motion_prediction','measured_pair_available':pair_measured,
                    'status':'independent_scene_observations' if pair_measured else ('unverified_camera_geometry' if not camera_verified else 'insufficient_independent_scene_observations')}
        records=[]
        for board in self.boards:
            state=temporary[board.board_id]
            self._states[board.board_id]=state
            record={k:v for k,v in state.items() if not k.startswith('_')}
            records.append(copy.deepcopy(record));self.history.append(copy.deepcopy(record))
        self._last_time=timestamp_ms;self._frame_ids.add(frame_id)
        return records
