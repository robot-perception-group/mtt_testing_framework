import os
import shutil
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def cleanup_log_files(log_dir=None, backup=True, older_than_days=None, extension='.log'):
    """
    Clean up log files in the specified directory and its subdirectories.
    
    Parameters:
    -----------
    log_dir : str or Path, optional
        Directory containing log files. If None, uses the 'logs' directory at the project root.
    backup : bool, default=True
        Whether to create a backup of deleted logs in a 'backup' folder.
    older_than_days : int, optional
        If provided, only remove files older than the specified number of days.
    extension : str, default='.log'
        File extension to identify log files.
        
    Returns:
    --------
    dict
        Statistics about the cleanup operation.
    """
    # Determine the logs directory
    if log_dir is None:
        # Get the current file's directory
        current_dir = Path(__file__).parent
        # Navigate to the logs directory (assumes logs are in a directory at project root)
        log_dir = current_dir.parent / 'logs'
    else:
        log_dir = Path(log_dir)
    
    if not log_dir.exists():
        logger.warning(f"Log directory {log_dir} does not exist!")
        return {"error": f"Directory {log_dir} not found", "files_removed": 0}
    
    # Create backup directory if needed
    backup_dir = None
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = log_dir / f"backup_{timestamp}"
        backup_dir.mkdir(exist_ok=True)
    
    # Statistics
    stats = {"files_removed": 0, "backup_location": str(backup_dir) if backup else None}
    
    # Find all log files
    log_files = []
    for file_path in log_dir.glob(f"**/*{extension}"):
        # Skip files in backup directories
        if "backup_" in str(file_path):
            continue
            
        # Check file age if needed
        if older_than_days is not None:
            file_age_days = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days
            if file_age_days < older_than_days:
                continue
                
        log_files.append(file_path)
    
    # Process each log file
    for file_path in log_files:
        try:
            if backup:
                # Create the same directory structure in the backup folder
                relative_path = file_path.relative_to(log_dir)
                backup_file_path = backup_dir / relative_path
                backup_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file to backup
                shutil.copy2(file_path, backup_file_path)
                
            # Remove original file
            file_path.unlink()
            logger.info(f"Removed log file: {file_path}")
            stats["files_removed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    # Remove empty directories
    for dirpath, dirnames, filenames in os.walk(log_dir, topdown=False):
        # Skip the backup directory
        if backup and Path(dirpath).is_relative_to(backup_dir):
            continue
            
        if not dirnames and not filenames:
            try:
                Path(dirpath).rmdir()
                logger.info(f"Removed empty directory: {dirpath}")
            except Exception as e:
                logger.error(f"Error removing directory {dirpath}: {e}")
    
    logger.info(f"Log cleanup completed. Removed {stats['files_removed']} files.")
    return stats
