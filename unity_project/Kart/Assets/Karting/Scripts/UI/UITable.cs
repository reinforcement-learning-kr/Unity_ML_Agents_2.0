using System;
using UnityEngine;


#if UNITY_EDITOR
using UnityEditor;

[CustomEditor(typeof(UITable), true)]
    
public class UITableEditor : Editor
{

    public override void OnInspectorGUI()
    {
        UITable myTarget = (UITable) target;
        DrawDefaultInspector();

        if (GUILayout.Button("Update"))
        {
            myTarget.UpdateTable(null);
        }
    }

}
#endif

public class UITable : MonoBehaviour
{

    [Tooltip("How much space should there be between items?")]    
    public float offset;

    [Tooltip("Add new the new items below existing items.")]
    public bool down;
    
    public void UpdateTable(GameObject newItem)
    {
        if (newItem != null) newItem.GetComponent<RectTransform>().localScale = Vector3.one;

        int childCount = transform.childCount;

        RectTransform hi = GetComponent<RectTransform>();

        float height = 0;
        for (int i = 0; i < childCount; i++)
        {
            RectTransform child = transform.GetChild(i).GetComponent<RectTransform>();
            Vector2 size = child.sizeDelta;
            height += down? -(1 - child.pivot.y) * size.y : (1 - child.pivot.y) * size.y;
            if (i != 0) height += down? -offset : offset;

            Vector2 newPos = Vector2.zero;
            
            newPos.y = height;
            newPos.x = 0;//-child.pivot.x * size.x * hi.localScale.x;
            child.anchoredPosition = newPos;
        }
    }
    
}
