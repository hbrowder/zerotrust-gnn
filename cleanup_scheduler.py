"""
GDPR Data Retention Cleanup Scheduler
Runs periodic cleanup of expired data based on retention policy
"""
import schedule
import time
import threading
from gdpr_consent import consent_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cleanup():
    """Execute data retention cleanup"""
    try:
        result = consent_manager.cleanup_expired_data()
        logger.info(f"Data retention cleanup completed: {result}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

def start_cleanup_scheduler():
    """
    Start background scheduler for data retention cleanup
    Runs daily at 2:00 AM
    """
    schedule.every().day.at("02:00").do(run_cleanup)
    
    def scheduler_loop():
        logger.info("GDPR data retention cleanup scheduler started (runs daily at 02:00)")
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
    scheduler_thread.start()
    logger.info("Cleanup scheduler thread started")

if __name__ == '__main__':
    logger.info("Starting GDPR cleanup scheduler as standalone process")
    run_cleanup()
    start_cleanup_scheduler()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Cleanup scheduler stopped")
