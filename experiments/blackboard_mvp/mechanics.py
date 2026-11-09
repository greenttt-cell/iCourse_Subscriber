"""Conservative physical sliding-board evidence tracking, entirely local.

Manual initial identities are required. Opposed motion is a noisy geometry prior,
never identity evidence. Hidden pixels never become observations or erasures.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np

from .core import BoardSpec, DetectorConfig, MultiBoardTracker, _gray, _coherent_shift
from .geometry import intersect, subtract


@dataclass(frozen=True)
class AssemblySpec:
    assembly_id: str
    viewport: tuple[int, int, int, int]

    def __post_init__(self):
        if not isinstance(self.assembly_id,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.assembly_id):
            raise ValueError("Invalid assembly_id")
        if len(self.viewport) != 4 or any(type(v) is not int for v in self.viewport):
            raise ValueError("Assembly viewport must be integer [x,y,width,height]")
        if min(self.viewport[:2]) < 0 or min(self.viewport[2:]) <= 0:
            raise ValueError("Invalid assembly viewport")

    def to_dict(self):
        return {"assembly_id": self.assembly_id, "viewport": list(self.viewport)}


def regions_from_mask(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [[int(v) for v in row[:4]] for row in stats[1:count]]


class SlidingBoardTracker:
    """Legacy/oracle regression tracker; automatic CLI uses DynamicMechanicalTracker.

    Track vertical translations with appearance evidence and manual checkpoints.

    Automatic matches require distinctive local features, a translation-consistent
    inlier set, and matching visible pixels. Unreliable tracks retain IDs as labels
    only; content detection stops until direct evidence or manual confirmation.
    """
    def __init__(self, boards: list[BoardSpec], assemblies: list[AssemblySpec], config=None):
        self.config = config or DetectorConfig()
        self.boards = list(boards)
        self.assemblies = {a.assembly_id: a for a in assemblies}
        if not boards or len({b.board_id for b in boards}) != len(boards):
            raise ValueError("Physical board IDs must be unique and nonempty")
        if not assemblies or len(self.assemblies) != len(assemblies):
            raise ValueError("Mechanical assemblies must be unique and nonempty")
        for aid in self.assemblies:
            members = [b for b in boards if b.assembly_id == aid]
            if len(members) != 2 or {b.layer for b in members} != {"outer", "inner"}:
                raise ValueError("Each assembly needs exactly one outer and one inner physical board")
        if any(b.assembly_id not in self.assemblies for b in boards):
            raise ValueError("Every physical board needs a configured assembly")
        self.events, self.versions, self.writing_runs, self.position_history = [], [], [], []
        self._states = {}
        self._frame_ids = set()
        self._previous_time = None
        self._previous_frame = None
        self._shape = None
        self._camera_invalid = False
        self._evidence = {}
        self._region_detector = MultiBoardTracker([BoardSpec('region_helper', 'comparison', (0,0,1,1))], self.config)

    def _visible(self, board, positions, shape):
        roi = positions[board.board_id]
        viewport = self.assemblies[board.assembly_id].viewport
        clipped = intersect(roi, viewport)
        clipped = intersect(clipped, [0, 0, shape[1], shape[0]]) if clipped else None
        parts = [clipped] if clipped else []
        if board.layer == 'inner':
            outer = next(b for b in self.boards if b.assembly_id == board.assembly_id and b.layer == 'outer')
            parts = [part for r in parts for part in subtract(r, positions[outer.board_id])]
        return parts

    @staticmethod
    def _canvas(frame, roi, visible):
        x, y, w, h = roi
        pixels = np.zeros((h,w,3), np.uint8)
        mask = np.zeros((h,w), np.uint8)
        for sx, sy, sw, sh in visible:
            pixels[sy-y:sy-y+sh, sx-x:sx-x+sw] = frame[sy:sy+sh, sx:sx+sw]
            mask[sy-y:sy-y+sh, sx-x:sx-x+sw] = 255
        return pixels, mask

    def _appearance(self, frame, board, old, roi, current_positions=None, strict=False):
        """Check known *visible* board pixels; never match a whole occluding crop."""
        positions = {b.board_id: list(self._states[b.board_id]['screen_roi']) if b.board_id in self._states else list(b.roi) for b in self.boards}
        positions.update(current_positions or {})
        positions[board.board_id] = roi
        visible = self._visible(board, positions, frame.shape)
        pixels, mask = self._canvas(frame, roi, visible)
        common = (old['known_mask'] > 0) & (mask > 0)
        if np.count_nonzero(common) < max(80, int(mask.size*.025)):
            return False
        diff = np.max(cv2.absdiff(pixels, old['reference']), axis=2)
        # A thin symbol may change while the surrounding distinctive texture still
        # anchors identity. Broad changes/teacher interference fail closed.
        if strict:
            return not np.any(diff[common] >= self.config.pixel_threshold)
        return float(np.mean(diff[common] < self.config.pixel_threshold)) >= .88

    def _estimate(self, frame, board, old, current_positions=None):
        roi = list(old['screen_roi'])
        reference_gray = _gray(old['reference'])
        orb = cv2.ORB_create(nfeatures=1000, edgeThreshold=7, patchSize=15, fastThreshold=7)
        reference_mask = cv2.erode(old['known_mask'], np.ones((3,3),np.uint8))
        kp1, d1 = orb.detectAndCompute(reference_gray, reference_mask)
        vx, vy, vw, vh = self.assemblies[board.assembly_id].viewport
        target = _gray(frame[vy:vy+vh, vx:vx+vw])
        kp2, d2 = orb.detectAndCompute(target, None)
        if d1 is not None and d2 is not None and len(d1) >= 8 and len(d2) >= 8:
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
            matches = [m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < .68*n.distance and m.distance <= 35]
            # Require independent target features; repeated strokes alone are weak.
            unique = {m.trainIdx:m for m in sorted(matches,key=lambda m:-m.distance)}
            matches = list(unique.values())
            if len(matches) >= 8:
                starts = np.array([kp1[m.queryIdx].pt for m in matches])
                ends = np.array([kp2[m.trainIdx].pt for m in matches]) + [vx,vy]
                translations = ends-starts
                median = np.median(translations,axis=0)
                inliers = np.linalg.norm(translations-median,axis=1) <= 1.6
                if inliers.sum() >= 8 and inliers.mean() >= .8 and abs(median[0]-board.roi[0]) <= 2:
                    candidate = [board.roi[0],int(round(median[1])),roi[2],roi[3]]
                    # Insufficient spatial support (e.g. one tiny repeated formula)
                    # cannot establish the identity of an entire moving panel.
                    spread = np.ptp(starts[inliers],axis=0)
                    if spread[0] >= min(20, roi[2]*.15) and spread[1] >= min(15,roi[3]*.12) and self._appearance(frame,board,old,candidate,current_positions):
                        return candidate, 'appearance_feature_translation', {'inliers':int(inliers.sum()),'matches':len(matches)}
        # Stationary direct appearance continuity is allowed only with distinctive
        # texture and a still-confirmed preceding position. No restore via blank
        # board similarity after an uncertain interval.
        known = old['known_mask'] > 0
        textured = np.count_nonzero(known) >= 80 and float(np.std(reference_gray[known])) >= 10
        if old['identity_status'] == 'confirmed' and textured and old['visibility'] != 'occluded' and self._appearance(frame,board,old,roi,current_positions,strict=True):
            return roi, 'stationary_appearance_continuity', {'inliers':None,'matches':None}
        return roi, 'unreliable', {'inliers':None,'matches':None}

    def _emit(self, board, frame_id, timestamp, crop_path, kind, old, reasons, uncertainty, **fields):
        e = {'event_id':f'{board.board_id}:{frame_id}:{kind}', 'board_id':board.board_id,
             'assembly_id':board.assembly_id, 'layer':board.layer, 'label':board.label, 'kind':kind,
             'observed_at_ms':timestamp, 'interval_ms':[old['timestamp_ms'] if old else timestamp,timestamp],
             'before_frame_id':old['frame_id'] if old else None,'after_frame_id':frame_id,
             'before_crop_path':old['crop_path'] if old else None,'after_crop_path':crop_path,
             'selected_reasons':reasons,'uncertainty':uncertainty,'change_regions':[],
             'version_id':old['version_id'] if old else None,
             'previous_version_id':old['version_id'] if old else None,
             'order_status':'observation_only','simultaneous_group':None, **fields}
        self.events.append(e)
        return e

    def _version(self, board, frame_id, timestamp, crop_path, old, status, visible):
        vid = f'{board.board_id}:v{sum(v["board_id"]==board.board_id for v in self.versions):04d}'
        self.versions.append({'version_id':vid,'board_id':board.board_id,'assembly_id':board.assembly_id,'layer':board.layer,
                             'frame_id':frame_id,'timestamp_ms':timestamp,'crop_path':crop_path,
                             'previous_version_id':old['version_id'] if old else None,'content_status':status,
                             'visibility':visible,'content_confirmation':'unverified_visual_observation',
                             'roi_status':'confirmed','not_a_complete_hidden_board_image':True})
        return vid

    def observe(self, frame, frame_id, timestamp_ms, image_path, crop_paths,
                time_status='verified_pts', force_reconfirm=False, manual_positions=None):
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError('Expected uint8 BGR frame')
        if type(timestamp_ms) is not int or timestamp_ms < 0 or (self._previous_time is not None and timestamp_ms <= self._previous_time):
            raise ValueError('Observation timestamps must increase strictly')
        if frame_id in self._frame_ids:
            raise ValueError('Frame IDs must be unique')
        manual_positions = manual_positions or {}
        if not isinstance(manual_positions,dict) or set(manual_positions)-{b.board_id for b in self.boards}:
            raise ValueError('Manual positions must identify configured physical boards')
        for bid, r in manual_positions.items():
            b = next(b for b in self.boards if b.board_id==bid)
            if not isinstance(r,(list,tuple)) or len(r)!=4 or any(type(v) is not int for v in r) or list(r[2:])!=list(b.roi[2:]) or r[0]!=b.roi[0]:
                raise ValueError('Manual checkpoint must preserve board size/x; changed camera geometry requires a fresh manually configured run')
        height,width=frame.shape[:2]
        if self._shape is None:
            if any(b.roi[2]>width*2 or b.roi[3]>height*2 or b.roi[2]*b.roi[3]>40_000_000 for b in self.boards):
                raise ValueError('Physical board dimensions exceed bounded prototype memory limits')
            for a in self.assemblies.values():
                x,y,w,h=a.viewport
                if x+w>width or y+h>height:
                    raise ValueError('Assembly viewport lies outside source frame')
            self._shape=frame.shape
        # Use only stationary camera-background pixels, never a whole-frame phase
        # score dominated by the mechanically moving panels.
        if frame.shape != self._shape or force_reconfirm:
            self._camera_invalid=True
        elif self._previous_frame is not None:
            outside=np.ones(frame.shape[:2],bool)
            for a in self.assemblies.values():
                x,y,w,h=a.viewport; outside[y:y+h,x:x+w]=False
            if np.count_nonzero(outside)>=100:
                changed=np.max(cv2.absdiff(frame,self._previous_frame),axis=2)>=self.config.pixel_threshold
                shift=_coherent_shift(_gray(self._previous_frame),_gray(frame),(outside.astype(np.uint8)*255))
                if np.mean(changed[outside])>.35 or (shift is not None and shift>self.config.motion_shift_pixels):
                    self._camera_invalid=True
        start=len(self.events)
        positions, sources, evidence, reliable = {}, {}, {}, {}
        for board in sorted(self.boards, key=lambda b: b.layer != "outer"):
            old=self._states.get(board.board_id)
            if board.board_id in manual_positions:
                positions[board.board_id]=list(manual_positions[board.board_id]); sources[board.board_id]='manual_checkpoint'; evidence[board.board_id]={}; reliable[board.board_id]=True
            elif old is None:
                positions[board.board_id]=list(board.roi); sources[board.board_id]='manual_initial_geometry'; evidence[board.board_id]={}; reliable[board.board_id]=True
            elif self._camera_invalid:
                positions[board.board_id]=list(old['screen_roi']); sources[board.board_id]='camera_geometry_unreliable'; evidence[board.board_id]={}; reliable[board.board_id]=False
            else:
                pos,source,proof=self._estimate(frame,board,old,positions)
                positions[board.board_id]=pos; sources[board.board_id]=source; evidence[board.board_id]=proof; reliable[board.board_id]=source!='unreliable'
        priors={}
        for aid in self.assemblies:
            outer=next(b for b in self.boards if b.assembly_id==aid and b.layer=='outer')
            inner=next(b for b in self.boards if b.assembly_id==aid and b.layer=='inner')
            oo,oi=self._states.get(outer.board_id),self._states.get(inner.board_id)
            if oo and oi:
                do=positions[outer.board_id][1]-oo['screen_roi'][1]
                di=positions[inner.board_id][1]-oi['screen_roi'][1]
                if reliable[outer.board_id] and not reliable[inner.board_id] and abs(do)>0:
                    positions[inner.board_id][1]=oi['screen_roi'][1]-do
                    sources[inner.board_id]='opposed_motion_prior_only'
                    di=-do
                residual=di+do if reliable[outer.board_id] and reliable[inner.board_id] else None
                priors[aid]={'outer_delta_y':do,'inner_delta_y':di,'residual_pixels':residual,
                             'tolerance_pixels':max(3,abs(do)*.15),
                             'status':'consistent' if residual is not None and abs(residual)<=max(3,abs(do)*.15) else ('inconsistent_requires_review' if residual is not None else 'unverified_prior_only'),
                             'identity_evidence':False}
                # The approximate linkage never overrules direct appearance or
                # a manual checkpoint. Inconsistency remains an explicit warning.
        frame_boards=[]
        for board in self.boards:
            bid=board.board_id; old=self._states.get(bid); roi=positions[bid]
            visible=self._visible(board,positions,frame.shape)
            pixels,mask=self._canvas(frame,roi,visible)
            fraction=float(np.count_nonzero(mask)/mask.size)
            visibility='occluded' if fraction==0 else ('visible' if fraction==1 else 'partially_occluded')
            # Uncertain outer geometry also makes the inner visibility mask unsafe.
            outer=next(b for b in self.boards if b.assembly_id==board.assembly_id and b.layer=='outer')
            certain=reliable[bid] and not self._camera_invalid and (board.layer=='outer' or reliable[outer.board_id])
            identity='confirmed' if certain else 'identity_uncertain'
            dy=roi[1]-old['screen_roi'][1] if old else 0
            moving=abs(dy)>0
            motion='moving' if moving and certain else ('unresolved' if not certain else 'stationary')
            crop_path=crop_paths.get(bid) if intersect(roi,[0,0,width,height]) else None
            vid=old['version_id'] if old else None
            reference=old['reference'].copy() if old else np.zeros_like(pixels)
            known=old['known_mask'].copy() if old else np.zeros_like(mask)
            baseline_evidence=old['baseline_evidence'] if old else None
            reference_ids=old['reference_ids'].copy() if old else np.full(mask.shape,-1,np.int32)
            sample_number=len(self._frame_ids)
            self._evidence.setdefault(bid,{})[sample_number]={'frame_id':frame_id,'crop_path':crop_path,'timestamp_ms':timestamp_ms}

            base_uncertainty=['visual_evidence_only_not_teaching_order']
            if time_status!='verified_pts': base_uncertainty.append('source_time_unverified')
            if old is None and fraction>0 and certain:
                vid=self._version(board,frame_id,timestamp_ms,crop_path,None,'initial_existing',visibility)
                self._emit(board,frame_id,timestamp_ms,crop_path,'initial_observation',None,['already_present_at_clip_start'],base_uncertainty+['appearance_time_before_clip_unknown'],version_id=vid)
                reference[mask>0]=pixels[mask>0];known[mask>0]=255;reference_ids[mask>0]=sample_number
                baseline_evidence={'frame_id':frame_id,'crop_path':crop_path,'timestamp_ms':timestamp_ms}
            if old and (moving or not certain):
                self._emit(board,frame_id,timestamp_ms,crop_path,'board_motion_event' if moving else 'identity_review_event',old,
                           ['physical_panel_translation' if moving else 'position_or_identity_unreliable'],base_uncertainty+([] if certain else ['identity_uncertain','manual_confirmation_required']),
                           delta_y=dy if certain else None,estimated_delta_y=dy,position_before=old['screen_roi'],position_after=roi,
                           identity_status=identity,tracking_source=sources[bid],tracking_evidence=evidence[bid],
                           compensation_status='translation_aligned_content_deferred_while_moving' if certain else 'unreliable_no_content_comparison',mechanical_prior=priors.get(board.assembly_id))
            if old and (visibility!=old['visibility'] or (certain and old['identity_status']!='confirmed')):
                self._emit(board,frame_id,timestamp_ms,crop_path,'visibility_event',old,['visibility_geometry_changed'],base_uncertainty+(['newly_revealed_content_timing_unknown'] if fraction>old['visible_fraction'] else []),
                           visibility_before=old['visibility'],visibility_after=visibility,visible_fraction=fraction,identity_status=identity,
                           reappearance_confirmed=certain and old['visibility']=='occluded' and fraction>0)
            # Full occlusion and unreliable identity never produce content events.
            # During movement, preserve samples and defer residual writing: do not
            # claim to separate a moving hand/stroke from mechanical displacement.
            if old and certain and fraction>0 and not moving:
                common=(known>0)&(mask>0)
                diff_before=reference.copy();diff_after=pixels.copy()
                diff_before[~common]=0;diff_after[~common]=0
                regions,count=self._region_detector._regions(diff_before,diff_after)
                compared=int(np.count_nonzero(common))
                if regions and compared and count/compared<self.config.interference_ratio:
                    # Evidence can span hidden/moving intervals. Never assign the
                    # change to the instant when old content becomes visible.
                    vid=self._version(board,frame_id,timestamp_ms,crop_path,old,'unverified_visible_content_change',visibility)
                    e=self._emit(board,frame_id,timestamp_ms,crop_path,'board_content_change_event',old,
                                 ['motion_compensated_shared_visible_pixels'],base_uncertainty+['change_time_within_evidence_interval_unknown','possible_teacher_interference_not_segmented'],
                                 version_id=vid,change_regions=regions,comparison_coordinate_space='physical_board_local',
                                 compared_visible_regions=regions_from_mask(common),compared_pixel_count=compared,
                                 compensation_status='translation_compensated',identity_status=identity)
                    changed_mask=(np.max(cv2.absdiff(diff_before,diff_after),axis=2)>=self.config.pixel_threshold)&common
                    refs=[]
                    for reference_id in sorted(np.unique(reference_ids[changed_mask])):
                        if reference_id>=0:
                            refs.append({**self._evidence[bid][int(reference_id)],
                                         'board_local_regions':regions_from_mask(changed_mask&(reference_ids==reference_id))})
                    if refs:
                        earliest=min(refs,key=lambda r:r['timestamp_ms'])
                        e.update(before_frame_id=earliest['frame_id'],before_crop_path=earliest['crop_path'],
                                 interval_ms=[earliest['timestamp_ms'],timestamp_ms],previous_sample_frame_id=old['frame_id'],
                                 comparison_references=refs,comparison_reference_semantics='per_pixel_last_confirmed_visible_evidence')
                        if len(refs)>1:
                            e['uncertainty'].append('multiple_historical_reference_frames_see_comparison_references')
                    reference[common]=pixels[common];reference_ids[common]=sample_number
                    baseline_evidence={'frame_id':frame_id,'crop_path':crop_path,'timestamp_ms':timestamp_ms}
                elif regions:
                    self._emit(board,frame_id,timestamp_ms,crop_path,'identity_review_event',old,
                               ['possible_teacher_interference_or_unresolved_content'],base_uncertainty+['manual_review_required'],
                               identity_status=identity,change_regions=regions,compensation_status='large_residual_not_classified_as_writing')
                # Newly visible pixels have no earlier observation. Store evidence
                # without claiming writing or erasure during the hidden interval.
                unseen=(mask>0)&(known==0)
                reference[unseen]=pixels[unseen];known[unseen]=255;reference_ids[unseen]=sample_number
                if baseline_evidence is None and np.any(unseen):
                    vid=self._version(board,frame_id,timestamp_ms,crop_path,old,'first_observed_after_occlusion',visibility)
                    baseline_evidence={'frame_id':frame_id,'crop_path':crop_path,'timestamp_ms':timestamp_ms}
            state={'board_id':bid,'assembly_id':board.assembly_id,'layer':board.layer,'label':board.label,
                   'frame_id':frame_id,'timestamp_ms':timestamp_ms,'crop_path':crop_path,
                   'screen_roi':list(roi),'crop_roi':intersect(roi,[0,0,width,height]),'visible_regions':visible,
                   'visible_fraction':fraction,'occlusion_ratio':1-fraction,'visibility':visibility,
                   'visibility_status':'geometry_confirmed' if certain else 'provisional_geometry_only',
                   'identity_status':identity,'roi_status':'confirmed' if certain else 'needs_reconfirmation',
                   'motion_status':motion,'tracking_source':sources[bid], 'version_id':vid,
                   'mechanical_prior':priors.get(board.assembly_id),
                   'comparison_status':'paused_occluded' if fraction==0 else ('paused_identity_uncertain' if not certain else ('deferred_during_motion' if moving else 'shared_visible_only'))}
            self.position_history.append(dict(state))
            self._states[bid]={**state,'reference':reference,'known_mask':known,'baseline_evidence':baseline_evidence,'reference_ids':reference_ids}
            frame_boards.append(state)
        new_events=self.events[start:]
        if len({e['board_id'] for e in new_events})>1:
            for e in new_events:
                e['simultaneous_group']=f'unordered:{frame_id}';e['order_status']='undetermined_within_interval'
                e['uncertainty'].append('cross_board_order_within_sample_unknown')
        self._frame_ids.add(frame_id);self._previous_time=timestamp_ms;self._previous_frame=frame.copy()
        return {'frame_id':frame_id,'timestamp_ms':timestamp_ms,'pts_time_ms':timestamp_ms,'time_status':time_status,
                'image_path':image_path,'boards':frame_boards,'event_ids':[e['event_id'] for e in new_events], 'selected':bool(new_events)}
