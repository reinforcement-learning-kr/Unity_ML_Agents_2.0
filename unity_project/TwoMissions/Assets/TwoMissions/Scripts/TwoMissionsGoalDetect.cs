using UnityEngine;
using UnityEngine.Events;

public class TwoMissionsGoalDetect : MonoBehaviour
{
    public UnityEvent ReachedGoal;
    void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.CompareTag("pushBallGoal"))
        {
            ReachedGoal.Invoke();
            Debug.Log("Scored!");
        }
    }
}
