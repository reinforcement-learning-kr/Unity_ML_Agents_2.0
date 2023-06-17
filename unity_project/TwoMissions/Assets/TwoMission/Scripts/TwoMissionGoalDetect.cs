using UnityEngine;
using UnityEngine.Events;

public class TwoMissionGoalDetect : MonoBehaviour
{
    public UnityEvent ReachedBallGoal;
    private void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.CompareTag("pushBallGoal"))
        {
            ReachedBallGoal.Invoke(); 
        }
    }
}
