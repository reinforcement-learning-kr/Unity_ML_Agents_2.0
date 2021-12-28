using UnityEngine;
 
    public class PoolInitializer : MonoBehaviour
    {
        public PoolObjectDef[] pools = new PoolObjectDef[0];
        public Transform parent;
        [Tooltip("Initialize all existing PoolObjects.")]
        

        private PoolObjectDef[] allPools;
        public void OnEnable()
        {
            initialize();
        }
        
        public void OnDisable()
        {            
            destroyAll();
        }

        public void initialize()
        {
            if (parent == null) parent = transform;

            destroyAll();

            for (int i = 0; i < pools.Length; i++) pools[i].Initialize(parent);

        }

        public void destroyAll()
        {
            for (int i = 0; i < pools.Length; i++) pools[i].destroyAll();
        }

    }
