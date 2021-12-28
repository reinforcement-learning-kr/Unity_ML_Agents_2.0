﻿using System.Collections;
 using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;


 [CreateAssetMenu(menuName = "Pool/Object")]
 public class PoolObjectDef : ScriptableObject
 {

	 [Tooltip("Spawn this many objects at the beginning of the game.")]
	 public int defaultSpawnCount = 5;
	 [Tooltip("Spawn this many objects in one go if the pool runs out of available objects.")]
	 public int makeExtraCount = 3;

	 [Tooltip("Object parent must be returned to reuse it. If false, object will be reused if disabled.")]
	 public bool returnToParentToReuse;

	 [Tooltip("The prefab that this pool will spawn")]
	 public GameObject poolObject;

	 private List<GameObject> pool = new List<GameObject>();
	 private Transform parent;
	 

	 public GameObject getObject(bool active, Transform parent)
	 {
		 if (!Application.isPlaying)
		 {
			 Debug.LogError($"PoolObject should not be used when the game is running.\n");
			 return null;
		 }

		 GameObject obj = null;
		 for (int i = 0; i < pool.Count; i++)
		 {
			 GameObject o = pool[i];
			 if (ReferenceEquals(o, null) || o.activeSelf) continue;
			 if (returnToParentToReuse && o.transform.parent.GetInstanceID() != this.parent.GetInstanceID()) continue;

			 obj = o;
			 break;
		 }


		 if (ReferenceEquals(obj, null))
		 {
			 if (pool.Count == 0)
			 {
				 Initialize(null);
				 obj = pool[0];
			 }
			 else
			 {
				 populateGroup(makeExtraCount);
				 obj = pool[pool.Count - makeExtraCount];
			 }
		 }

		 if (parent) obj.transform.SetParent(parent);
		 obj.SetActive(active);

		 return obj;
	 }

	 public void returnAll()
	 {
		 for (int i = 0; i < pool.Count; i++)
		 {
			 for (int j = 0; j < pool.Count; j++) returnObject(pool[i], true);
		 }
	 }

	 public void destroyAll()
	 {

		 bool isPlaying = Application.isPlaying;


		 for (int i = pool.Count - 1; i >= 0; i--)
			 if (isPlaying)
				 Destroy(pool[i]);
			 else
				 DestroyImmediate(pool[i]);

		 pool = new List<GameObject>();
	 }

	 public void returnObject(GameObject obj, bool returnToParent)
	 {
		 if (returnToParentToReuse) returnToParent = true;
		 
		 if (parent && returnToParent) obj.transform.SetParent(parent);
		 obj.SetActive(false);
	 }

	 public void Initialize(Transform parent)
	 {
		 Assert.IsNotNull(poolObject, $"{name} is missing its poolObject!\n");
		 this.parent = parent;
		 populateGroup(defaultSpawnCount);
	 }

	 private void populateGroup(int need)
	 {
		// Debug.Log($"!!!!! {name} POPULATE {need}\n");
		 while (need > 0)
		 {

			 GameObject o = Instantiate(poolObject);
			 o.name = o.name + "_" + pool.Count;
			 if (parent) o.transform.SetParent(parent, false);
			 o.SetActive(false);
			 pool.Add(o);
			 need--;
		 }
	 }

	 public IEnumerator ReturnWithDelay(GameObject obj, float delay)
	 {
		 yield return new WaitForSeconds(delay);
		 returnObject(obj,true);
	 }
	 
	 
	 public IEnumerator DisableWithDelay(GameObject obj, float delay)
	 {
		 yield return new WaitForSeconds(delay);
		 obj.SetActive(false);
	 }
 }

