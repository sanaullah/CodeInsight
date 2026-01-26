"""
Centralized logging configuration for the Framework.

Provides JSON-formatted or readable file logging with automatic rollover,
date-based file naming, module-specific loggers, and config.yaml integration.
"""

import json
import logging
import logging.handlers
import os
import sys
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


class TraceContextFilter(logging.Filter):
    """Filter that adds trace context (trace_id, user_id, session_id) to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace context to log record."""
        try:
            from utils.langfuse.tracing import get_current_trace_context
            context = get_current_trace_context()
            record.trace_id = context.get("trace_id")
            record.user_id = context.get("user_id")
            record.session_id = context.get("session_id")
        except (ImportError, AttributeError, Exception):
            # Silently fail if tracing is not available or context cannot be retrieved
            # This ensures logging continues to work even if Langfuse is disabled
            record.trace_id = None
            record.user_id = None
            record.session_id = None
        return True  # Always allow record through


class TraceAwareFormatter(logging.Formatter):
    """Formatter that includes trace context in console output when available."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Get base formatted message
        base_msg = super().format(record)
        
        # Append trace context if available
        trace_parts = []
        if hasattr(record, "trace_id") and record.trace_id:
            trace_parts.append(f"trace_id={record.trace_id}")
        if hasattr(record, "user_id") and record.user_id:
            trace_parts.append(f"user_id={record.user_id}")
        if hasattr(record, "session_id") and record.session_id:
            trace_parts.append(f"session_id={record.session_id}")
        
        if trace_parts:
            return f"{base_msg} [{', '.join(trace_parts)}]"
        return base_msg


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add optional fields if available
        if hasattr(record, "module") and record.module:
            log_data["module"] = record.module
        elif hasattr(record, "pathname") and record.pathname:
            # Extract module name from pathname
            try:
                module_name = Path(record.pathname).stem
                log_data["module"] = module_name
            except Exception:
                pass
        
        if hasattr(record, "funcName") and record.funcName:
            log_data["function"] = record.funcName
        
        if hasattr(record, "lineno") and record.lineno:
            log_data["line"] = record.lineno
        
        # Add correlation ID if present
        if hasattr(record, "correlation_id") and record.correlation_id:
            log_data["correlation_id"] = record.correlation_id
        
        # Add trace context if present
        if hasattr(record, "trace_id") and record.trace_id:
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "user_id") and record.user_id:
            log_data["user_id"] = record.user_id
        if hasattr(record, "session_id") and record.session_id:
            log_data["session_id"] = record.session_id
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class WindowsSafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """
    Windows-safe timed rotating file handler that handles file locking issues.
    
    On Windows, file operations can fail when files are still open. This handler
    closes the file before rotation and uses copy + delete instead of rename
    for better Windows compatibility.
    """
    
    def emit(self, record):
        """
        Emit a record, handling permission errors gracefully.
        
        If the log file is locked or inaccessible, silently skip file logging
        to prevent the application from crashing.
        """
        try:
            # If stream is None (file logging disabled due to permission error), skip
            if self.stream is None:
                # Try to reopen the file
                try:
                    if not self.delay:
                        self.stream = self._open()
                except (OSError, PermissionError):
                    # Still can't open, skip this log entry
                    return
            
            super().emit(record)
        except (OSError, PermissionError):
            # If we get a permission error during emit, disable file logging
            # and silently skip this log entry
            self.stream = None
            # Don't try to log this error to avoid recursion
            return
        except Exception:
            self.handleError(record)
    
    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        
        Windows-safe implementation that handles file locking gracefully.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Get current time for rotation
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        
        # Calculate new rollover time
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
        
        if os.path.exists(dfn):
            try:
                os.remove(dfn)
            except (OSError, PermissionError):
                pass
        
        if os.path.exists(self.baseFilename):
            try:
                if sys.platform == 'win32':
                    # On Windows, use copy + delete approach
                    try:
                        shutil.copy2(self.baseFilename, dfn)
                    except (OSError, PermissionError):
                        # If copy fails, wait and retry once
                        time.sleep(0.1)
                        try:
                            shutil.copy2(self.baseFilename, dfn)
                        except (OSError, PermissionError):
                            # Still locked, skip rotation this time
                            pass
                    # Try to truncate the original file
                    try:
                        with open(self.baseFilename, 'w', encoding=self.encoding):
                            pass  # Truncate file
                    except (OSError, PermissionError):
                        pass
                else:
                    # On Unix-like systems, use standard rename
                    os.rename(self.baseFilename, dfn)
            except (OSError, PermissionError):
                # If rotation fails, skip this time
                pass
        
        # Update rollover time for next rotation (from parent class logic)
        if self.utc:
            t = time.time()
        else:
            t = time.time()
        newRolloverAt = self.computeRollover(t)
        while newRolloverAt <= t:
            newRolloverAt = newRolloverAt + self.interval
        self.rolloverAt = newRolloverAt
        
        # Reopen the file
        if not self.delay:
            try:
                self.stream = self._open()
            except (OSError, PermissionError):
                # If we can't reopen the file, silently continue
                self.stream = None
                pass


class WindowsSafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Windows-safe rotating file handler that handles file locking issues.
    
    On Windows, os.rename() fails when the file is still open. This handler
    closes the file before rotation and uses copy + delete instead of rename
    for better Windows compatibility.
    """
    
    def emit(self, record):
        """
        Emit a record, handling permission errors gracefully.
        
        If the log file is locked or inaccessible, silently skip file logging
        to prevent the application from crashing.
        """
        try:
            # If stream is None (file logging disabled due to permission error), skip
            if self.stream is None:
                # Try to reopen the file
                try:
                    if not self.delay:
                        self.stream = self._open()
                except (OSError, PermissionError):
                    # Still can't open, skip this log entry
                    return
            
            super().emit(record)
        except (OSError, PermissionError):
            # If we get a permission error during emit, disable file logging
            # and silently skip this log entry
            self.stream = None
            # Don't try to log this error to avoid recursion
            return
        except Exception:
            self.handleError(record)
    
    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        
        Windows-safe implementation that handles file locking gracefully.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            # Delete the oldest backup if it exists
            s = self.baseFilename + '.' + str(self.backupCount)
            if os.path.exists(s):
                try:
                    os.remove(s)
                except (OSError, PermissionError):
                    # If we can't delete it, log a warning but continue
                    pass
            
            # Rotate existing backups
            for i in range(self.backupCount - 1, 0, -1):
                sfn = self.baseFilename + '.' + str(i)
                dfn = self.baseFilename + '.' + str(i + 1)
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        try:
                            os.remove(dfn)
                        except (OSError, PermissionError):
                            pass
                    try:
                        # Use copy instead of rename on Windows for better compatibility
                        if sys.platform == 'win32':
                            shutil.copy2(sfn, dfn)
                            # Try to remove source, but don't fail if locked
                            try:
                                os.remove(sfn)
                            except (OSError, PermissionError):
                                # File might still be locked, that's okay
                                pass
                        else:
                            os.rename(sfn, dfn)
                    except (OSError, PermissionError) as e:
                        # If rotation fails, log but don't crash
                        # The file will be rotated on next attempt
                        pass
            
            # Rotate the current log file
            dfn = self.baseFilename + '.1'
            if os.path.exists(self.baseFilename):
                try:
                    if sys.platform == 'win32':
                        # On Windows, use copy + truncate approach
                        # First, try to copy the file
                        try:
                            shutil.copy2(self.baseFilename, dfn)
                        except (OSError, PermissionError):
                            # If copy fails, the file might be locked
                            # Wait a bit and try once more
                            time.sleep(0.1)
                            try:
                                shutil.copy2(self.baseFilename, dfn)
                            except (OSError, PermissionError):
                                # Still locked, skip rotation this time
                                # The file will be rotated on next attempt
                                pass
                        # Try to truncate the original file
                        try:
                            with open(self.baseFilename, 'w', encoding=self.encoding):
                                pass  # Truncate file
                        except (OSError, PermissionError):
                            # Can't truncate, that's okay - next rotation will handle it
                            pass
                    else:
                        # On Unix-like systems, use standard rename
                        os.rename(self.baseFilename, dfn)
                except (OSError, PermissionError) as e:
                    # If rotation fails, log but don't crash
                    # The file will be rotated on next attempt
                    pass
        
        # Reopen the file
        if not self.delay:
            try:
                self.stream = self._open()
            except (OSError, PermissionError) as e:
                # If we can't reopen the file (e.g., locked by another process),
                # silently continue without file logging
                # This prevents the entire application from crashing due to log file issues
                # The file might be locked by another process or antivirus software
                # We'll try again on the next log write
                self.stream = None
                # Don't log this error to avoid recursion - just silently fail
                pass


def cleanup_old_logs(
    log_dir: Path,
    retention_days: int = 30,
    date_based_files: bool = True,
    backup_count: int = 5
) -> None:
    """
    Remove log files older than retention_days.
    
    Args:
        log_dir: Root log directory
        retention_days: Number of days to keep logs (default: 30)
        date_based_files: Whether using date-based file naming
        backup_count: Number of rotated backup files to keep per day
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_timestamp = cutoff_date.timestamp()
        
        # Scan all subdirectories
        for subdir in log_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Process files in subdirectory
            for log_file in subdir.iterdir():
                if not log_file.is_file() or not log_file.suffix == '.log':
                    continue
                
                # Check if file is old based on modification time
                file_mtime = log_file.stat().st_mtime
                
                if file_mtime < cutoff_timestamp:
                    # File is older than retention period
                    try:
                        log_file.unlink()
                    except (OSError, PermissionError):
                        # If we can't delete it, skip it
                        pass
                elif date_based_files:
                    # For date-based files, also check filename for date
                    # Format: app_YYYY-MM-DD.log or app_YYYY-MM-DD.log.N
                    try:
                        # Extract date from filename
                        name_parts = log_file.stem.split('_')
                        if len(name_parts) >= 2:
                            # Try to parse date from filename
                            date_str = name_parts[-1].split('.')[0]  # Get date part before .N
                            file_date = datetime.strptime(date_str, "%Y-%m-%d")
                            if file_date < cutoff_date:
                                try:
                                    log_file.unlink()
                                except (OSError, PermissionError):
                                    pass
                    except (ValueError, IndexError):
                        # If we can't parse the date, use modification time
                        if file_mtime < cutoff_timestamp:
                            try:
                                log_file.unlink()
                            except (OSError, PermissionError):
                                pass
    except Exception:
        # Silently fail cleanup to avoid breaking logging setup
        pass


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    log_file: str = "app.log",
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 5,
    console_output: bool = True,
    use_json_format: bool = False,
    date_based_files: bool = False,
    module_loggers: Optional[List[str]] = None,
    enable_time_rotation: bool = False,
    retention_days: int = 30
) -> None:
    """
    Set up centralized logging configuration.
    
    Args:
        log_level: Logging level (default: logging.INFO)
        log_dir: Directory for log files (default: project_root/logs)
        log_file: Name of log file (default: app.log)
        max_bytes: Maximum log file size before rollover (default: 5MB)
        backup_count: Number of backup log files to keep (default: 5)
        console_output: Whether to output logs to console (default: True)
        use_json_format: Whether to use JSON format for file logs (default: False)
        date_based_files: Whether to use date-based file naming (default: False)
        module_loggers: List of module names for separate log files (default: None)
        enable_time_rotation: Whether to enable time-based (daily) rotation (default: False)
        retention_days: Number of days to keep log files (default: 30)
    """
    # Determine log directory - always resolve relative to project root
    project_root = Path(__file__).parent.parent
    if log_dir is None:
        log_dir = project_root / "logs"
    else:
        log_dir = Path(log_dir)
        # If relative path, resolve against project root
        if not log_dir.is_absolute():
            log_dir = project_root / log_dir
    
    # Create logs directory structure
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "app").mkdir(exist_ok=True)
    (log_dir / "errors").mkdir(exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Determine file name
    today = datetime.now().strftime("%Y-%m-%d")
    if date_based_files:
        app_log_file = log_dir / "app" / f"app_{today}.log"
        error_log_file = log_dir / "errors" / f"errors_{today}.log"
    else:
        app_log_file = log_dir / "app" / log_file
        error_log_file = log_dir / "errors" / f"errors_{log_file}"
    
    # Create formatters
    if use_json_format:
        file_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_formatter = TraceAwareFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create application log file handler with Windows-safe rotation
    # Note: With date_based_files=True, the filename already includes the date,
    # so we use size-based rotation within each day. Time-based rotation is
    # more useful when NOT using date-based files.
    if enable_time_rotation and not date_based_files:
        # Use time-based rotation for daily rotation (when not using date-based filenames)
        file_handler = WindowsSafeTimedRotatingFileHandler(
            filename=str(app_log_file),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
    else:
        # Use size-based rotation (works with both date-based and non-date-based files)
        file_handler = WindowsSafeRotatingFileHandler(
            filename=str(app_log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(TraceContextFilter())
    root_logger.addHandler(file_handler)
    
    # Create error-only log file handler with Windows-safe rotation
    if enable_time_rotation and not date_based_files:
        error_handler = WindowsSafeTimedRotatingFileHandler(
            filename=str(error_log_file),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
    else:
        error_handler = WindowsSafeRotatingFileHandler(
            filename=str(error_log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    error_handler.addFilter(TraceContextFilter())
    root_logger.addHandler(error_handler)
    
    # Create console handler for terminal output
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(TraceContextFilter())
        root_logger.addHandler(console_handler)
    
    # Setup module-specific loggers if requested
    if module_loggers:
        for module_name in module_loggers:
            _setup_module_logger(
                module_name,
                log_dir,
                today if date_based_files else None,
                max_bytes,
                backup_count,
                file_formatter,
                enable_time_rotation and date_based_files
            )
    
    # Cleanup old log files
    if retention_days > 0:
        cleanup_old_logs(log_dir, retention_days, date_based_files, backup_count)
    
    # Log initialization message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging initialized: app_file={app_log_file}, error_file={error_log_file}, "
        f"max_bytes={max_bytes}, backup_count={backup_count}, json_format={use_json_format}, "
        f"time_rotation={enable_time_rotation}, retention_days={retention_days}"
    )


def _setup_module_logger(
    module_name: str,
    log_dir: Path,
    date_str: Optional[str],
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    use_time_rotation: bool = False
) -> None:
    """Setup a module-specific logger with its own file handler."""
    # Create module-specific log directory
    module_log_dir = log_dir / module_name.replace(".", "_")
    module_log_dir.mkdir(exist_ok=True)
    
    # Determine file name
    if date_str:
        log_file = module_log_dir / f"{module_name.replace('.', '_')}_{date_str}.log"
    else:
        log_file = module_log_dir / f"{module_name.replace('.', '_')}.log"
    
    # Get module logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    
    # Create handler with Windows-safe rotation
    # Note: With date-based files, use size-based rotation since filename already includes date
    if use_time_rotation and not date_str:
        handler = WindowsSafeTimedRotatingFileHandler(
            str(log_file),
            when='midnight',
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
    else:
        handler = WindowsSafeRotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    handler.addFilter(TraceContextFilter())
    logger.addHandler(handler)
    logger.propagate = True  # Also propagate to root logger


def setup_logging_from_config() -> None:
    """
    Set up logging from config.yaml using ConfigManager.
    
    This function reads logging configuration from config.yaml and initializes
    logging accordingly. If logging is disabled or config is missing, it will
    use sensible defaults.
    """
    try:
        from llm.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Check if logging config exists
        if not hasattr(config, 'logging'):
            # No logging config, use defaults
            setup_logging()
            return
        
        logging_config = config.logging
        
        # Check if logging is enabled
        if not logging_config.enabled:
            return
        
        # Convert log level string to int
        log_level_str = logging_config.log_level.upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        # Get rotation config
        rotation = logging_config.rotation
        max_bytes = rotation.get("max_bytes", 10 * 1024 * 1024)  # 10MB default
        backup_count = rotation.get("backup_count", 5)
        retention_days = rotation.get("retention_days", 30)
        enable_time_rotation = rotation.get("enable_time_rotation", True)
        
        # Get module loggers config
        module_loggers = None
        if hasattr(logging_config, 'module_loggers') and logging_config.module_loggers:
            if isinstance(logging_config.module_loggers, dict) and logging_config.module_loggers.get("enabled"):
                module_loggers = logging_config.module_loggers.get("modules", [])
            elif isinstance(logging_config.module_loggers, list):
                module_loggers = logging_config.module_loggers
        
        # Resolve log_dir path relative to project root
        project_root = Path(__file__).parent.parent
        log_dir_str = logging_config.log_dir if hasattr(logging_config, 'log_dir') else None
        if log_dir_str:
            log_dir_path = Path(log_dir_str)
            if not log_dir_path.is_absolute():
                # Resolve relative to project root
                log_dir = project_root / log_dir_path
            else:
                log_dir = log_dir_path
        else:
            log_dir = None
        
        # Setup logging
        setup_logging(
            log_level=log_level,
            log_dir=log_dir,
            log_file="app.log",
            max_bytes=max_bytes,
            backup_count=backup_count,
            console_output=logging_config.enable_console_logging if hasattr(logging_config, 'enable_console_logging') else True,
            use_json_format=logging_config.use_json_format if hasattr(logging_config, 'use_json_format') else False,
            date_based_files=logging_config.date_based_files if hasattr(logging_config, 'date_based_files') else True,
            module_loggers=module_loggers,
            enable_time_rotation=enable_time_rotation,
            retention_days=retention_days
        )
        
    except Exception as e:
        # Don't fail if logging setup fails, just warn
        import sys
        print(f"Warning: Failed to setup logging from config: {e}", file=sys.stderr)


