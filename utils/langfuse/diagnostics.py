"""
Langfuse diagnostic and testing utilities.
"""

import os
import logging
from typing import Dict, Any
from .client import get_langfuse_client
from .connection import validate_langfuse_connection

logger = logging.getLogger(__name__)


def diagnose_langfuse_litellm_integration() -> Dict[str, Any]:
    """
    Comprehensive diagnostic function to test Langfuse-LiteLLM integration.
    
    Returns:
        Dictionary with diagnostic results including:
        - langfuse_client_initialized: bool
        - environment_variables_set: bool
        - litellm_callbacks_configured: bool
        - connection_test_passed: bool
        - test_trace_sent: bool
        - errors: List[str]
    """
    diagnostics = {
        "langfuse_client_initialized": False,
        "environment_variables_set": False,
        "litellm_callbacks_configured": False,
        "connection_test_passed": False,
        "test_trace_sent": False,
        "errors": []
    }
    
    try:
        # Test 1: Check Langfuse client initialization
        logger.info("🔍 Diagnostic Test 1: Langfuse Client Initialization")
        client = get_langfuse_client()
        if client is None:
            diagnostics["errors"].append("Langfuse client not initialized")
            logger.error("❌ Langfuse client not initialized")
        else:
            diagnostics["langfuse_client_initialized"] = True
            logger.info("✅ Langfuse client initialized")
        
        # Test 2: Check environment variables
        logger.info("🔍 Diagnostic Test 2: Environment Variables")
        required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        # Check for OTEL host (preferred) or regular host (fallback)
        otel_host = os.environ.get("LANGFUSE_OTEL_HOST")
        regular_host = os.environ.get("LANGFUSE_HOST")
        
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            diagnostics["errors"].append(f"Missing environment variables: {', '.join(missing_vars)}")
            logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        elif not otel_host and not regular_host:
            diagnostics["errors"].append("Missing LANGFUSE_OTEL_HOST or LANGFUSE_HOST")
            logger.error("❌ Missing LANGFUSE_OTEL_HOST or LANGFUSE_HOST")
        else:
            diagnostics["environment_variables_set"] = True
            logger.info("✅ All required environment variables are set")
            if otel_host:
                logger.debug(f"LANGFUSE_OTEL_HOST: {otel_host}")
            if regular_host:
                logger.debug(f"LANGFUSE_HOST: {regular_host}")
            logger.debug(f"LANGFUSE_PUBLIC_KEY: {os.environ.get('LANGFUSE_PUBLIC_KEY')[:10]}...")
        
        # Test 3: Check LiteLLM callbacks
        logger.info("🔍 Diagnostic Test 3: LiteLLM Callbacks")
        try:
            import litellm
            # Check for langfuse_otel callback (OTEL integration)
            callbacks = getattr(litellm, 'callbacks', [])
            if "langfuse_otel" in callbacks:
                diagnostics["litellm_callbacks_configured"] = True
                logger.info("✅ LiteLLM callbacks configured with langfuse_otel (OTEL integration)")
                logger.debug(f"LiteLLM callbacks: {callbacks}")
            elif hasattr(litellm, 'success_callback') and "langfuse" in str(litellm.success_callback):
                # Fallback check for older callback style
                diagnostics["litellm_callbacks_configured"] = True
                logger.info("✅ LiteLLM callbacks configured for Langfuse (legacy)")
                logger.debug(f"LiteLLM success_callback: {litellm.success_callback}")
            else:
                diagnostics["errors"].append("LiteLLM callbacks not configured for Langfuse")
                logger.error("❌ LiteLLM callbacks not configured for Langfuse")
                logger.debug(f"Current callbacks: {callbacks}")
                if hasattr(litellm, 'success_callback'):
                    logger.debug(f"Current success_callback: {litellm.success_callback}")
        except ImportError:
            diagnostics["errors"].append("LiteLLM not installed")
            logger.error("❌ LiteLLM not installed")
        except Exception as e:
            diagnostics["errors"].append(f"Error checking LiteLLM callbacks: {str(e)}")
            logger.error(f"❌ Error checking LiteLLM callbacks: {e}")
        
        # Test 4: Connection test
        logger.info("🔍 Diagnostic Test 4: Langfuse Connection")
        if client:
            if validate_langfuse_connection():
                diagnostics["connection_test_passed"] = True
                logger.info("✅ Langfuse connection test passed")
            else:
                diagnostics["errors"].append("Langfuse connection test failed")
                logger.error("❌ Langfuse connection test failed")
        
        # Test 5: Send a test trace using v3 API
        logger.info("🔍 Diagnostic Test 5: Send Test Trace")
        if client:
            try:
                # Create test trace using v3 start_as_current_observation
                with client.start_as_current_observation(
                    as_type="span",
                    name="diagnostic_test_trace",
                    input={"test": True, "diagnostic": True},
                    metadata={"test": True, "diagnostic": True}
                ) as test_trace:
                    # Update trace with output
                    test_trace.update(
                        output="Test trace completed successfully",
                        metadata={"status": "success"}
                    )
                    # Explicitly set trace input/output
                    test_trace.update_trace(
                        input={"test": True, "diagnostic": True},
                        output="Test trace completed successfully"
                    )
                
                client.flush()
                diagnostics["test_trace_sent"] = True
                logger.info("✅ Test trace sent successfully")
            except Exception as e:
                diagnostics["errors"].append(f"Failed to send test trace: {str(e)}")
                logger.error(f"❌ Failed to send test trace: {e}")
        
        # Summary
        all_passed = all([
            diagnostics["langfuse_client_initialized"],
            diagnostics["environment_variables_set"],
            diagnostics["litellm_callbacks_configured"],
            diagnostics["connection_test_passed"],
            diagnostics["test_trace_sent"]
        ])
        
        diagnostics["all_checks_passed"] = all_passed
        
        if all_passed:
            logger.info("🎉 All diagnostic tests passed!")
        else:
            logger.warning(f"⚠️ Some diagnostic tests failed. Errors: {diagnostics['errors']}")
        
        return diagnostics
        
    except Exception as e:
        diagnostics["errors"].append(f"Diagnostic function error: {str(e)}")
        logger.error(f"❌ Diagnostic function error: {e}", exc_info=True)
        return diagnostics

