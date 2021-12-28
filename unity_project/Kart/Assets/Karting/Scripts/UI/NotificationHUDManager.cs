using UnityEngine;

public class NotificationHUDManager : MonoBehaviour
{
    [Tooltip("UI panel containing the layoutGroup for displaying notifications")]
    public UITable notificationPanel;
    [Tooltip("Prefab for the notifications")]
    public PoolObjectDef notificationPrefab;
    

    void OnUpdateObjective(UnityActionUpdateObjective updateObjective)
    {
        if (!string.IsNullOrEmpty(updateObjective.notificationText))
            CreateNotification(updateObjective.notificationText);
    }

    public void CreateNotification(string text)
    {
        GameObject notificationInstance = notificationPrefab.getObject(true,notificationPanel.transform);
        notificationInstance.transform.SetSiblingIndex(0);

        NotificationToast toast = notificationInstance.GetComponent<NotificationToast>();
        toast.Initialize(text);
        notificationPanel.UpdateTable(notificationInstance);

    }

    public void RegisterObjective(Objective objective)
    {
        objective.onUpdateObjective += OnUpdateObjective;
    }

    public void UnregisterObjective(Objective objective)
    {
        objective.onUpdateObjective -= OnUpdateObjective;
    }
}
