"""
Streamlit verification UI for hard negative mining human-in-the-loop workflow.

This module implements the human verification interface as specified in the
Code_Structure.txt documentation. It provides an efficient UI for confirming
false positive detections in the hard negative mining cycle.

Key Features:
- Streamlit-based web interface for image review
- Efficient keyboard shortcuts and pagination
- Atomic logging of confirmed hard negatives
- Progress tracking and session management
- Real-time queue statistics and updates

References:
- Code_Structure.txt: Detailed app.py specifications
- Streamlit best practices for scientific applications
- Human-computer interaction principles for ML annotation

Author: Foundational Flower Detector Team
Date: September 2025
"""

import os
import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
from PIL import Image
import cv2

# Project imports
from ..config import Config
from ..data_preparation.utils import AtomicFileWriter

# Configure logging for Streamlit
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationApp:
    """
    Streamlit application for human verification of false positives.
    
    This class implements the verification UI functionality specified in the
    Code_Structure.txt documentation. It provides an efficient interface for
    human reviewers to confirm false positive detections in the hard negative
    mining workflow.
    
    The UI follows human-computer interaction principles for efficient
    annotation workflows with keyboard shortcuts and batch processing.
    """
    
    def __init__(self):
        """Initialize verification application."""
        self.config = None
        self.ui_config = {}
        self.verification_queue = None
        self.queue_file_path = None
        self.confirmed_log_path = None
        self.session_log_path = None
        
        # UI state
        self.current_page = 0
        self.items_per_page = 10
        self.total_items = 0
        
        # Load configuration
        self._load_config()
        
        # Setup paths
        self._setup_paths()
        
        logger.info("Verification app initialized")
    
    def _load_config(self):
        """Load configuration and UI settings."""
        try:
            self.config = Config()
            self.ui_config = self.config.get('ui', {})
            
            # UI settings
            self.items_per_page = self.ui_config.get('images_per_page', 10)
            
        except Exception as e:
            st.error(f"Failed to load configuration: {e}")
            logger.error(f"Config loading failed: {e}")
    
    def _setup_paths(self):
        """Setup file paths for verification workflow."""
        if self.config:
            data_paths = self.config.get_data_paths()
            mining_config = self.config.get('hard_negative_mining', {})
            
            self.queue_file_path = data_paths['base'] / mining_config.get(
                'verification_queue_file', 'verification_queue.json'
            )
            self.confirmed_log_path = data_paths['base'] / mining_config.get(
                'confirmed_negatives_log', 'confirmed_hard_negatives.log'
            )
            self.session_log_path = data_paths['base'] / 'verification_session.log'
    
    def load_verification_queue(self) -> Optional[Dict[str, Any]]:
        """
        Load verification queue from JSON file.
        
        This function implements the load_verification_queue functionality
        specified in Code_Structure.txt, reading the queue created by the
        hard negative mining process.
        
        Returns:
            Verification queue data or None if not found
        """
        if not self.queue_file_path or not self.queue_file_path.exists():
            return None
        
        try:
            with open(self.queue_file_path, 'r', encoding='utf-8') as f:
                queue_data = json.load(f)
            
            logger.info(f"Loaded verification queue with {queue_data.get('total_items', 0)} items")
            return queue_data
            
        except Exception as e:
            logger.error(f"Failed to load verification queue: {e}")
            return None
    
    def display_image_for_review(self, item: Dict[str, Any], item_index: int) -> Optional[str]:
        """
        Display image and controls for human review.
        
        This function implements the core UI functionality specified in
        Code_Structure.txt, displaying images with bounding boxes and
        providing confirmation controls.
        
        Args:
            item: False positive item to review
            item_index: Index of item in queue
            
        Returns:
            User decision ('confirm', 'reject', None)
        """
        image_path = Path(item['image_path'])
        confidence = item.get('confidence', 0.0)
        
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image
            if image_path.exists():
                try:
                    # Load and display image
                    image = self._load_and_display_image(image_path, item)
                    
                except Exception as e:
                    st.error(f"Failed to load image: {e}")
                    return None
            else:
                st.error(f"Image not found: {image_path}")
                return None
        
        with col2:
            # Display metadata
            st.subheader("Detection Info")
            st.write(f"**File:** {image_path.name}")
            st.write(f"**Confidence:** {confidence:.3f}")
            st.write(f"**Model:** {item.get('model_version', 'Unknown')}")
            st.write(f"**Detected:** {item.get('scan_timestamp', 'Unknown')}")
            
            # Review instructions
            st.markdown("---")
            st.subheader("Review Instructions")
            st.write("""
            **Question:** Is this truly a FALSE POSITIVE?
            
            ✅ **Confirm**: This is NOT a flower (false positive)
            ❌ **Reject**: This IS a flower (correct detection)
            """)
            
            # Decision buttons
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("✅ Confirm FP", key=f"confirm_{item_index}", 
                           help="This is NOT a flower (false positive)"):
                    return 'confirm'
            
            with col_b:
                if st.button("❌ Reject FP", key=f"reject_{item_index}",
                           help="This IS a flower (correct detection)"):
                    return 'reject'
            
            # Skip option
            if st.button("⏭️ Skip", key=f"skip_{item_index}",
                        help="Skip this item for now"):
                return 'skip'
        
        return None
    
    def _load_and_display_image(self, image_path: Path, item: Dict[str, Any]) -> np.ndarray:
        """
        Load image and display with bounding box overlay.
        
        Args:
            image_path: Path to image file
            item: Detection item with bbox info
            
        Returns:
            Loaded image array
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw bounding box if available
        if 'bbox' in item:
            bbox = item['bbox']
            if isinstance(bbox, dict):
                x = int(bbox.get('x', 0))
                y = int(bbox.get('y', 0))
                w = int(bbox.get('width', image.shape[1]))
                h = int(bbox.get('height', image.shape[0]))
                
                # Draw rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                
                # Add confidence label
                confidence = item.get('confidence', 0.0)
                label = f"Flower: {confidence:.2f}"
                cv2.putText(image, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Resize for display
        display_size = self.ui_config.get('max_image_display_size', [800, 600])
        height, width = image.shape[:2]
        
        # Calculate scaling to fit display size
        scale_w = display_size[0] / width
        scale_h = display_size[1] / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Display in Streamlit
        st.image(image, caption=f"Detection: {image_path.name}", use_column_width=True)
        
        return image
    
    def append_to_log(self, item: Dict[str, Any], decision: str):
        """
        Append confirmed hard negative to log file.
        
        This function implements the atomic logging functionality specified
        in Code_Structure.txt, safely recording confirmed false positives.
        
        Args:
            item: False positive item
            decision: Human decision ('confirm', 'reject', 'skip')
        """
        if decision != 'confirm':
            return  # Only log confirmed false positives
        
        try:
            # Create log entry
            log_entry = {
                'image_path': item['image_path'],
                'confidence': item.get('confidence', 0.0),
                'model_version': item.get('model_version', 'unknown'),
                'scan_timestamp': item.get('scan_timestamp'),
                'confirmation_timestamp': datetime.now().isoformat(),
                'confirmed_by': 'human_reviewer',
                'verification_session': st.session_state.get('session_id', 'unknown')
            }
            
            # Append to confirmed negatives log
            with AtomicFileWriter.atomic_write(self.confirmed_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            logger.info(f"Confirmed hard negative logged: {item['image_path']}")
            
        except Exception as e:
            logger.error(f"Failed to log confirmed negative: {e}")
            st.error(f"Failed to save confirmation: {e}")
    
    def _log_session_activity(self, activity: Dict[str, Any]):
        """Log session activity for tracking and analysis."""
        try:
            activity['timestamp'] = datetime.now().isoformat()
            activity['session_id'] = st.session_state.get('session_id', 'unknown')
            
            with AtomicFileWriter.atomic_write(self.session_log_path, 'a') as f:
                f.write(json.dumps(activity) + '\n')
                
        except Exception as e:
            logger.warning(f"Failed to log session activity: {e}")
    
    def display_progress_stats(self, queue_data: Dict[str, Any]):
        """Display progress statistics and queue information."""
        if not queue_data:
            return
        
        # Get progress info
        progress = queue_data.get('verification_progress', {})
        total_items = queue_data.get('total_items', 0)
        confirmed = progress.get('confirmed', 0)
        rejected = progress.get('rejected', 0)
        pending = progress.get('pending', total_items)
        
        # Calculate percentages
        completed = confirmed + rejected
        completion_rate = (completed / total_items * 100) if total_items > 0 else 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Items", total_items)
        with col2:
            st.metric("Confirmed FPs", confirmed)
        with col3:
            st.metric("Rejected", rejected)
        with col4:
            st.metric("Completion", f"{completion_rate:.1f}%")
        
        # Progress bar
        if total_items > 0:
            progress_bar = st.progress(completion_rate / 100)
            st.caption(f"{completed}/{total_items} items reviewed ({pending} pending)")
        
        # Queue metadata
        with st.expander("Queue Information"):
            st.write(f"**Created:** {queue_data.get('created_at', 'Unknown')}")
            st.write(f"**Model Version:** {queue_data.get('model_version', 'Unknown')}")
            st.write(f"**Confidence Threshold:** {queue_data.get('confidence_threshold', 'Unknown')}")
            
            if 'ui_metadata' in queue_data:
                ui_meta = queue_data['ui_metadata']
                st.write(f"**Instructions:** {ui_meta.get('instructions', '')}")
    
    def run_verification_interface(self):
        """
        Main verification interface orchestration.
        
        This function implements the main application loop specified in
        Code_Structure.txt, managing session state, image presentation,
        and user interactions.
        """
        # Page configuration
        st.set_page_config(
            page_title=self.ui_config.get('title', 'Flower Detector Verification'),
            page_icon=self.ui_config.get('page_icon', '🌸'),
            layout=self.ui_config.get('layout', 'wide'),
            initial_sidebar_state=self.ui_config.get('initial_sidebar_state', 'expanded')
        )
        
        # Initialize session state
        if 'session_id' not in st.session_state:
            st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state.items_reviewed = 0
            st.session_state.items_confirmed = 0
            st.session_state.items_rejected = 0
            st.session_state.current_item_index = 0
        
        # Main title
        st.title("🌸 Foundational Flower Detector")
        st.subheader("False Positive Verification Interface")
        
        # Load verification queue
        queue_data = self.load_verification_queue()
        
        if not queue_data:
            st.warning("No verification queue found. Run hard negative mining first.")
            st.info("Command: `flower-detector-mine-negatives`")
            return
        
        # Display progress stats
        self.display_progress_stats(queue_data)
        
        # Get items to review
        false_positives = queue_data.get('false_positives', [])
        
        if not false_positives:
            st.success("🎉 All items in the verification queue have been processed!")
            return
        
        # Pagination controls
        total_items = len(false_positives)
        
        # Sidebar controls
        with st.sidebar:
            st.header("Navigation")
            
            # Page navigation
            max_pages = (total_items - 1) // self.items_per_page + 1
            page_num = st.number_input(
                "Page", 
                min_value=1, 
                max_value=max_pages, 
                value=st.session_state.get('current_page', 1)
            )
            st.session_state.current_page = page_num - 1
            
            st.write(f"Page {page_num} of {max_pages}")
            st.write(f"Items {page_num * self.items_per_page - self.items_per_page + 1}-"
                    f"{min(page_num * self.items_per_page, total_items)} of {total_items}")
            
            # Session statistics
            st.header("Session Stats")
            st.metric("Items Reviewed", st.session_state.items_reviewed)
            st.metric("Confirmed FPs", st.session_state.items_confirmed)
            st.metric("Rejected", st.session_state.items_rejected)
            
            # Actions
            st.header("Actions")
            if st.button("🔄 Refresh Queue"):
                st.experimental_rerun()
            
            if st.button("📊 Export Session Log"):
                self._export_session_log()
            
            # Keyboard shortcuts info
            st.header("Keyboard Shortcuts")
            st.caption("(when supported by browser)")
            st.write("- `Enter`: Confirm FP")
            st.write("- `Space`: Reject FP") 
            st.write("- `→`: Next item")
            st.write("- `←`: Previous item")
        
        # Main content area
        st.markdown("---")
        
        # Calculate current page items
        start_idx = st.session_state.current_page * self.items_per_page
        end_idx = min(start_idx + self.items_per_page, total_items)
        current_items = false_positives[start_idx:end_idx]
        
        # Process items on current page
        for i, item in enumerate(current_items):
            item_index = start_idx + i
            
            st.markdown(f"### Item {item_index + 1} of {total_items}")
            
            # Display item for review
            decision = self.display_image_for_review(item, item_index)
            
            if decision:
                # Process decision
                self._process_decision(item, decision, item_index)
                
                # Update session stats
                st.session_state.items_reviewed += 1
                if decision == 'confirm':
                    st.session_state.items_confirmed += 1
                elif decision == 'reject':
                    st.session_state.items_rejected += 1
                
                # Log session activity
                self._log_session_activity({
                    'action': 'item_reviewed',
                    'item_index': item_index,
                    'decision': decision,
                    'image_path': item['image_path'],
                    'confidence': item.get('confidence', 0.0)
                })
                
                # Show feedback
                if decision == 'confirm':
                    st.success("✅ Confirmed as false positive")
                elif decision == 'reject':
                    st.info("❌ Rejected - marked as correct detection")
                elif decision == 'skip':
                    st.warning("⏭️ Skipped for now")
                
                # Auto-advance option
                if self.ui_config.get('auto_advance', True) and decision != 'skip':
                    time.sleep(1)  # Brief pause for feedback
                    if item_index < total_items - 1:
                        st.experimental_rerun()
            
            st.markdown("---")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.current_page > 0:
                if st.button("← Previous Page"):
                    st.session_state.current_page -= 1
                    st.experimental_rerun()
        
        with col3:
            if st.session_state.current_page < max_pages - 1:
                if st.button("Next Page →"):
                    st.session_state.current_page += 1
                    st.experimental_rerun()
        
        # Auto-refresh option
        if self.ui_config.get('queue_refresh_interval', 0) > 0:
            refresh_interval = self.ui_config['queue_refresh_interval']
            st.write(f"Auto-refresh every {refresh_interval} seconds")
            time.sleep(refresh_interval)
            st.experimental_rerun()
    
    def _process_decision(self, item: Dict[str, Any], decision: str, item_index: int):
        """Process user decision for an item."""
        if decision == 'confirm':
            # Log confirmed false positive
            self.append_to_log(item, decision)
            
        # Update queue file to track progress
        self._update_queue_progress(item_index, decision)
    
    def _update_queue_progress(self, item_index: int, decision: str):
        """Update verification queue with progress information."""
        try:
            # Load current queue
            if self.queue_file_path.exists():
                with open(self.queue_file_path, 'r') as f:
                    queue_data = json.load(f)
                
                # Update progress
                progress = queue_data.setdefault('verification_progress', {
                    'confirmed': 0, 'rejected': 0, 'pending': queue_data.get('total_items', 0)
                })
                
                if decision == 'confirm':
                    progress['confirmed'] += 1
                elif decision == 'reject':
                    progress['rejected'] += 1
                
                progress['pending'] = max(0, progress['pending'] - 1)
                
                # Mark item as processed
                if 'false_positives' in queue_data and item_index < len(queue_data['false_positives']):
                    queue_data['false_positives'][item_index]['verification_status'] = decision
                    queue_data['false_positives'][item_index]['verification_timestamp'] = datetime.now().isoformat()
                
                # Save updated queue
                with AtomicFileWriter.atomic_write(self.queue_file_path) as f:
                    json.dump(queue_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update queue progress: {e}")
    
    def _export_session_log(self):
        """Export session log for download."""
        if self.session_log_path and self.session_log_path.exists():
            try:
                with open(self.session_log_path, 'r') as f:
                    log_content = f.read()
                
                st.download_button(
                    label="Download Session Log",
                    data=log_content,
                    file_name=f"verification_session_{st.session_state.session_id}.log",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"Failed to export session log: {e}")


def main():
    """
    Main entry point for verification UI.
    
    This function can be called directly or used as a command-line script
    for launching the verification interface.
    """
    try:
        # Create and run verification app
        app = VerificationApp()
        app.run_verification_interface()
        
    except Exception as e:
        st.error(f"Verification app failed: {e}")
        logger.error(f"App failure: {e}")


if __name__ == "__main__":
    main()
