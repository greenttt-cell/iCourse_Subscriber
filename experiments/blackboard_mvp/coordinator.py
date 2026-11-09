"""S0–S2 automatic geometry coordinator; ordinary content admission stays OFF."""
from __future__ import annotations

import copy
import cv2
import numpy as np

from .mechanics import SlidingBoardTracker, intersect, subtract
from .camera import CameraMonitor
from .tracking import DynamicBoardTracker
from .models import BoardGeometry, BoardIdentity, comparison_gate


class DynamicMechanicalTracker(SlidingBoardTracker):
    """Reuse legacy evidence helpers, not its camera/identity/content state flow."""
    def __init__(self, boards, assemblies, config=None, camera_config=None, physical_apertures=None):
        super().__init__(boards, assemblies, config)
        options = dict(camera_config or {})
        self.camera = CameraMonitor(excluded_regions=[list(a.viewport) for a in assemblies], **options)
        self.tracker = DynamicBoardTracker(boards, assemblies)
        self.apertures = physical_apertures or {}
        self.camera_history = []
        self.aggregation_status = 'not_implemented_in_s0_s2'

    def observe(self, frame, frame_id, timestamp_ms, image_path, crop_paths,
                time_status='verified_pts', force_reconfirm=False, manual_positions=None):
        if manual_positions:
            raise ValueError('Automatic tracking rejects future manual positions; use explicit oracle_geometry mode')
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected uint8 BGR frame')
        if type(timestamp_ms) is not int or timestamp_ms < 0 or (self._previous_time is not None and timestamp_ms <= self._previous_time):
            raise ValueError('Observation timestamps must increase strictly')
        if frame_id in self._frame_ids:
            raise ValueError('Frame IDs must be unique')
        if not self._frame_ids:
            for a in self.assemblies.values():
                x,y,w,h=a.viewport
                if x+w>frame.shape[1] or y+h>frame.shape[0]:raise ValueError('Assembly search region outside source frame')
            for aperture in self.apertures.values():
                if aperture is not None and (len(aperture)!=4 or any(type(v) is not int for v in aperture) or min(aperture[:2])<0 or min(aperture[2:])<=0 or aperture[0]+aperture[2]>frame.shape[1] or aperture[1]+aperture[3]>frame.shape[0]):
                    raise ValueError('Physical aperture outside source frame')
        camera = self.camera.observe(frame, frame_id, timestamp_ms, force_reconfirm=force_reconfirm)
        self.camera_history.append(copy.deepcopy(camera))
        observations = self.tracker.observe(frame, frame_id, timestamp_ms, camera)
        observed = {s['board_id']: s for s in observations}
        start = len(self.events)
        states = []
        for board in self.boards:
            bid = board.board_id
            raw = copy.deepcopy(observed[bid])
            old = self._states.get(bid)
            roi = raw.get('measured_screen_roi')
            predicted = raw.get('predicted_screen_roi')
            evidence_roi = roi or predicted or raw.get('last_verified_screen_roi')
            confirmed = raw.get('identity_status') == 'confirmed' and camera['reliability'] == 'verified' and roi is not None
            cropped = intersect(evidence_roi, [0, 0, frame.shape[1], frame.shape[0]]) if evidence_roi else None
            visible = []
            outer = next(b for b in self.boards if b.assembly_id == board.assembly_id and b.layer == 'outer')
            outer_state = observed[outer.board_id]
            # Only observed geometry can define known mechanical regions. A missing
            # physical aperture is not silently replaced by the search viewport.
            aperture = self.apertures.get(board.assembly_id)
            geometry_known = confirmed and aperture is not None
            if geometry_known:
                clipped = intersect(roi, aperture)
                clipped = intersect(clipped, [0, 0, frame.shape[1], frame.shape[0]]) if clipped else None
                visible = [clipped] if clipped else []
                if board.layer == 'inner':
                    if outer_state.get('identity_status') == 'confirmed' and outer_state.get('measured_screen_roi'):
                        visible = [p for r in visible for p in subtract(r, outer_state['measured_screen_roi'])]
                    else:
                        visible = []
                        geometry_known = False
            fraction = sum(r[2]*r[3] for r in visible)/(roi[2]*roi[3]) if geometry_known and roi else None
            visibility = 'unknown' if fraction is None else ('occluded' if fraction == 0 else ('visible' if fraction == 1 else 'partially_occluded'))
            state = {**raw, 'board_id':bid, 'assembly_id':board.assembly_id, 'layer':board.layer, 'label':board.label,
                     'frame_id':frame_id, 'timestamp_ms':timestamp_ms, 'screen_roi':roi,
                     'evidence_roi':evidence_roi, 'evidence_roi_source':'measurement' if roi is not None else 'prediction_or_last_known',
                     'crop_roi':cropped, 'crop_path':crop_paths.get(bid) if cropped else None,
                     'visible_regions':visible, 'visible_fraction':fraction, 'fraction_semantics':'initial_observed_fragment_only', 'full_physical_board_visible_fraction':None, 'occlusion_ratio':None if fraction is None else 1-fraction,
                     'visibility':visibility, 'visibility_status':'geometry_confirmed_without_person_segmentation' if geometry_known else 'unknown_geometry',
                     'camera_reliability':camera['reliability'], 'reason_codes':[raw.get('tracking_source','unavailable')]+([] if camera['reliability']=='verified' else ['scene_geometry_unverified']), 'physical_extent_status':'unknown', 'scene_alignment_status':'verified' if camera['reliability']=='verified' else 'unavailable', 'roi_status':'observed_screen_geometry' if roi is not None else 'unavailable',
                     'comparison_status':'disabled_s0_s2',
                     'comparison_eligibility':{'eligible':False,'reason_codes':['content_admission_disabled_s0_s2','foreground_and_fine_registration_not_implemented']},
                     'baseline_id':old.get('baseline_id') if old else None,
                     'version_id':old.get('version_id') if old else None}
            reset = confirmed and (old is None or not old.get('baseline_id') or old.get('identity_status') != 'confirmed' or old.get('comparison_epoch') != state.get('comparison_epoch'))
            if reset:
                vid = self._version(board,frame_id,timestamp_ms,state['crop_path'],old,
                                    'initial_existing' if old is None else 'reacquired_geometry_baseline_not_content_change',visibility)
                state['version_id'] = vid
                state['baseline_id'] = vid
                self._emit(board,frame_id,timestamp_ms,state['crop_path'],'content_baseline_initialized',old,
                           ['initial_manual_identity' if old is None else 'new_comparison_epoch_after_reacquisition'],
                           ['geometry_evidence_only','content_admission_disabled_s0_s2','unobserved_changes_not_inferred'],
                           version_id=vid, comparison_epoch=state.get('comparison_epoch'))
            if old and old.get('identity_status') != state.get('identity_status'):
                self._emit(board,frame_id,timestamp_ms,state['crop_path'],
                           'identity_recovered' if state['identity_status']=='confirmed' else 'identity_lost',old,
                           list(state.get('reason_codes',[])),['identity_separate_from_geometry'],
                           identity_status=state['identity_status'], tracking_evidence=state.get('tracking_evidence'))
            if old and roi is not None and old.get('measured_screen_roi') is not None and confirmed and raw.get('motion_status') == 'moving' and raw.get('delta_scene_px') is not None:
                previous = old['measured_screen_roi']
                if np.linalg.norm(raw['delta_scene_px']) > .75:
                    self._emit(board,frame_id,timestamp_ms,state['crop_path'],'board_motion_event',old,
                               ['direct_visual_geometry'],['not_a_writing_event'],
                               delta_y=raw['delta_scene_px'][1], delta_scene_px=raw['delta_scene_px'], screen_delta_y=roi[1]-previous[1], position_before=previous,position_after=roi,
                               compensation_status='content_disabled_s0_s2',tracking_evidence=state.get('tracking_evidence'))
            if not confirmed:
                self._emit(board,frame_id,timestamp_ms,state['crop_path'],'identity_review_event' if state['identity_status']!='confirmed' else 'geometry_review_event',old,
                           list(state.get('reason_codes',[])) or ['insufficient_independent_evidence'],
                           (['identity_uncertain'] if state['identity_status']!='confirmed' else ['camera_geometry_uncertain','scene_motion_unresolved'])+['no_content_comparison'], estimated_position=predicted,
                           tracking_evidence=state.get('tracking_evidence'))
            geometry_contract = BoardGeometry(bid,frame_id,raw['measurement_status'],roi,predicted,
                raw.get('prediction_uncertainty_px'),
                {'roi':raw.get('last_verified_screen_roi'),'frame_id':raw.get('last_verified_frame_id'),'timestamp_ms':raw.get('last_verified_timestamp_ms')},
                raw['motion_status'],raw.get('geometry_epoch',0)).to_dict()
            identity_contract = BoardIdentity(bid,'confirmed' if raw['identity_status']=='confirmed' else 'uncertain',
                raw.get('identity_epoch',0),'manual_initial' if old is None else ('direct_visual' if raw['identity_status']=='confirmed' else None),
                raw.get('identity_confidence',{})).to_dict()
            state['geometry_contract']=geometry_contract
            state['identity_contract']=identity_contract
            state['comparison_eligibility']=comparison_gate(camera_reliability=camera['reliability'],
                identity_status=state['identity_status'],measurement_status=raw['measurement_status'],
                baseline_ready=state['baseline_id'] is not None,content_enabled=False,visibility_status=visibility,
                comparison_epoch=raw.get('comparison_epoch',0),baseline_id=state['baseline_id'])
            state['camera_transform_ref']=camera.get('transform_ref')
            state['transform_ref']=camera.get('transform_ref')
            state['transform_ref_semantics']='camera_screen_to_scene_not_complete_board_homography'
            state['identity_transform_link_verified']=confirmed
            self.position_history.append(copy.deepcopy(state))
            self._states[bid] = copy.deepcopy(state)
            states.append(state)
        self._frame_ids.add(frame_id)
        self._previous_time = timestamp_ms
        return {'frame_id':frame_id,'timestamp_ms':timestamp_ms,'pts_time_ms':timestamp_ms,'time_status':time_status,
                'image_path':image_path,'boards':states,'camera':camera,
                'event_ids':[e['event_id'] for e in self.events[start:]],'selected':bool(self.events[start:])}
