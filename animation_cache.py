import os
import json
import hashlib
from pathlib import Path
import logging
from s3_storage import S3Storage

logger = logging.getLogger(__name__)

class AnimationCache:
    def __init__(self, s3_storage=None):
        self.s3_storage = s3_storage or S3Storage()
        self.cache_dir = Path("./outputs/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self):
        if self.cache_index_path.exists():
            try:
                with open(self.cache_index_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache index: {e}")
                return {}
        return {}
    
    def _save_cache_index(self):
        try:
            with open(self.cache_index_path, "w") as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")
    
    def _generate_cache_key(self, prompt):
        return hashlib.md5(prompt.lower().strip().encode()).hexdigest()
    
    def get_cached_animation(self, prompt):
        cache_key = self._generate_cache_key(prompt)
        
        if cache_key in self.cache_index:
            cache_entry = self.cache_index[cache_key]
            logger.info(f"Cache hit for prompt: {prompt[:30]}...")
            
            if self.s3_storage.is_enabled and "s3_url" in cache_entry:
                return {
                    "id": cache_entry.get("job_id"),
                    "title": cache_entry.get("title"),
                    "s3_url": cache_entry.get("s3_url"),
                    "cached": True
                }
            elif "local_path" in cache_entry:
                local_path = cache_entry["local_path"]
                if os.path.exists(local_path):
                    return {
                        "id": cache_entry.get("job_id"),
                        "title": cache_entry.get("title"),
                        "local_path": local_path,
                        "cached": True
                    }
        
        logger.info(f"Cache miss for prompt: {prompt[:30]}...")
        return None
    
    def cache_animation(self, prompt, job_id, title, video_path=None, s3_url=None):
        cache_key = self._generate_cache_key(prompt)
        
        cache_entry = {
            "prompt": prompt,
            "job_id": job_id,
            "title": title
        }
        
        if video_path:
            cache_entry["local_path"] = video_path
            
        if s3_url:
            cache_entry["s3_url"] = s3_url
            
        self.cache_index[cache_key] = cache_entry
        self._save_cache_index()
        logger.info(f"Cached animation for prompt: {prompt[:30]}...")
        
    def preload_common_animations(self, animations_data):
        for animation in animations_data:
            prompt = animation.get("prompt")
            job_id = animation.get("job_id")
            title = animation.get("title")
            s3_key = animation.get("s3_key")
            
            if not all([prompt, job_id, title, s3_key]):
                logger.warning(f"Missing required data for preloading animation: {animation}")
                continue
                
            s3_url = self.s3_storage.get_url(s3_key)
            if s3_url:
                self.cache_animation(prompt, job_id, title, s3_url=s3_url)
                logger.info(f"Preloaded animation: {title}")
            else:
                logger.warning(f"Failed to preload animation: {title}")
