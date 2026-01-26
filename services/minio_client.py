"""
MinIO Client Wrapper.

Provides connection management, bucket operations, file upload/download,
error handling, and health checks.
"""

import logging
from typing import Optional, BinaryIO, Dict, Any
from pathlib import Path
from services.db_config import get_db_config, MinIOConfig

logger = logging.getLogger(__name__)

# Try to import boto3, but don't fail if not available
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False
    logger.warning("boto3 not installed. Install with: pip install boto3")


class MinIOClient:
    """
    MinIO client wrapper with connection management.
    
    Provides:
    - Connection management
    - Bucket operations
    - File upload/download
    - Error handling
    - Health checks
    """
    
    def __init__(self, config: Optional[MinIOConfig] = None):
        """
        Initialize MinIO client.
        
        Args:
            config: Optional MinIO configuration (defaults to from_env)
        """
        if not MINIO_AVAILABLE:
            raise ImportError("boto3 is not installed. Install with: pip install boto3")
        
        self.config = config or get_db_config().minio
        self._s3_client = None
        self._connected = False
    
    def _get_client(self):
        """
        Get or create S3/MinIO client.
        
        Returns:
            S3 client instance
        """
        if self._s3_client is None or not self._connected:
            try:
                s3_config = self.config.get_s3_config()
                self._s3_client = boto3.client(
                    's3',
                    **s3_config
                )
                # Test connection by listing buckets
                self._s3_client.list_buckets()
                self._connected = True
                logger.info(f"MinIO client connected to {self.config.endpoint}")
            except Exception as e:
                logger.error(f"Failed to connect to MinIO: {e}")
                self._connected = False
                raise
        
        return self._s3_client
    
    def ensure_bucket(self, bucket: Optional[str] = None) -> None:
        """
        Ensure bucket exists, create if it doesn't.
        
        Args:
            bucket: Bucket name (defaults to config bucket)
        """
        bucket_name = bucket or self.config.bucket
        client = self._get_client()
        
        try:
            # Check if bucket exists
            client.head_bucket(Bucket=bucket_name)
            logger.debug(f"Bucket {bucket_name} already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Bucket doesn't exist, create it
                try:
                    client.create_bucket(Bucket=bucket_name)
                    logger.info(f"Created bucket {bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket {bucket_name}: {create_error}")
                    raise
            else:
                logger.error(f"Failed to check bucket {bucket_name}: {e}")
                raise
    
    def upload_file(self, file_path: str, object_name: Optional[str] = None, bucket: Optional[str] = None) -> str:
        """
        Upload a file to MinIO.
        
        Args:
            file_path: Path to local file
            object_name: Object name in bucket (defaults to filename)
            bucket: Bucket name (defaults to config bucket)
            
        Returns:
            Object name in bucket
        """
        bucket_name = bucket or self.config.bucket
        self.ensure_bucket(bucket_name)
        
        if object_name is None:
            object_name = Path(file_path).name
        
        client = self._get_client()
        
        try:
            client.upload_file(file_path, bucket_name, object_name)
            logger.debug(f"Uploaded {file_path} to {bucket_name}/{object_name}")
            return object_name
        except Exception as e:
            logger.error(f"Failed to upload {file_path}: {e}")
            raise
    
    def upload_fileobj(self, file_obj: BinaryIO, object_name: str, bucket: Optional[str] = None) -> str:
        """
        Upload a file-like object to MinIO.
        
        Args:
            file_obj: File-like object
            object_name: Object name in bucket
            bucket: Bucket name (defaults to config bucket)
            
        Returns:
            Object name in bucket
        """
        bucket_name = bucket or self.config.bucket
        self.ensure_bucket(bucket_name)
        
        client = self._get_client()
        
        try:
            client.upload_fileobj(file_obj, bucket_name, object_name)
            logger.debug(f"Uploaded file object to {bucket_name}/{object_name}")
            return object_name
        except Exception as e:
            logger.error(f"Failed to upload file object: {e}")
            raise
    
    def download_file(self, object_name: str, file_path: str, bucket: Optional[str] = None) -> None:
        """
        Download a file from MinIO.
        
        Args:
            object_name: Object name in bucket
            file_path: Path to save file
            bucket: Bucket name (defaults to config bucket)
        """
        bucket_name = bucket or self.config.bucket
        client = self._get_client()
        
        try:
            client.download_file(bucket_name, object_name, file_path)
            logger.debug(f"Downloaded {bucket_name}/{object_name} to {file_path}")
        except Exception as e:
            logger.error(f"Failed to download {object_name}: {e}")
            raise
    
    def download_fileobj(self, object_name: str, file_obj: BinaryIO, bucket: Optional[str] = None) -> None:
        """
        Download a file from MinIO to a file-like object.
        
        Args:
            object_name: Object name in bucket
            file_obj: File-like object to write to
            bucket: Bucket name (defaults to config bucket)
        """
        bucket_name = bucket or self.config.bucket
        client = self._get_client()
        
        try:
            client.download_fileobj(bucket_name, object_name, file_obj)
            logger.debug(f"Downloaded {bucket_name}/{object_name} to file object")
        except Exception as e:
            logger.error(f"Failed to download {object_name}: {e}")
            raise
    
    def delete_file(self, object_name: str, bucket: Optional[str] = None) -> None:
        """
        Delete a file from MinIO.
        
        Args:
            object_name: Object name in bucket
            bucket: Bucket name (defaults to config bucket)
        """
        bucket_name = bucket or self.config.bucket
        client = self._get_client()
        
        try:
            client.delete_object(Bucket=bucket_name, Key=object_name)
            logger.debug(f"Deleted {bucket_name}/{object_name}")
        except Exception as e:
            logger.error(f"Failed to delete {object_name}: {e}")
            raise
    
    def file_exists(self, object_name: str, bucket: Optional[str] = None) -> bool:
        """
        Check if a file exists in MinIO.
        
        Args:
            object_name: Object name in bucket
            bucket: Bucket name (defaults to config bucket)
            
        Returns:
            True if file exists, False otherwise
        """
        bucket_name = bucket or self.config.bucket
        client = self._get_client()
        
        try:
            client.head_object(Bucket=bucket_name, Key=object_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Failed to check file existence: {e}")
            return False
    
    def get_file_url(self, object_name: str, bucket: Optional[str] = None, expires_in: int = 3600) -> str:
        """
        Get a presigned URL for a file.
        
        Args:
            object_name: Object name in bucket
            bucket: Bucket name (defaults to config bucket)
            expires_in: URL expiration time in seconds
            
        Returns:
            Presigned URL
        """
        bucket_name = bucket or self.config.bucket
        client = self._get_client()
        
        try:
            url = client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': object_name},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise
    
    def list_files(self, prefix: str = "", bucket: Optional[str] = None) -> list[str]:
        """
        List files in a bucket.
        
        Args:
            prefix: Object name prefix to filter
            bucket: Bucket name (defaults to config bucket)
            
        Returns:
            List of object names
        """
        bucket_name = bucket or self.config.bucket
        client = self._get_client()
        
        try:
            response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check MinIO connection health.
        
        Returns:
            Dictionary with health status
        """
        try:
            client = self._get_client()
            client.list_buckets()
            return {
                "status": "healthy",
                "connected": True,
                "endpoint": self.config.endpoint,
                "bucket": self.config.bucket,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "endpoint": self.config.endpoint,
            }
    
    def close(self) -> None:
        """Close MinIO connection."""
        # boto3 clients don't need explicit closing
        self._s3_client = None
        self._connected = False
        logger.debug("MinIO connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Global client instance
_client: Optional[MinIOClient] = None


def get_minio_client() -> MinIOClient:
    """
    Get or create global MinIO client instance.
    
    Returns:
        MinIOClient instance
    """
    global _client
    if _client is None:
        _client = MinIOClient()
    return _client

