using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongGoalDetection : MonoBehaviour
{
    public PongAgent AgentA;
    public PongAgent AgentB;

    private Rigidbody RbBall;
    
    private Vector3 Resetpos;
    private Vector3 velocity;

    private Transform ballTrans;

    private float max_ball_speed = 10f;
    private float min_ball_speed = 5f;

    void Start()
    {
        RbBall = gameObject.GetComponent<Rigidbody>();
        ballTrans = transform;
        Resetpos = transform.position;
    }

    public void ResetPosition()
    {
        ballTrans.position = Resetpos;
        RbBall.velocity = Vector3.zero;
        RbBall.angularVelocity = Vector3.zero;
        ballTrans.rotation = Quaternion.identity;

        float rand_num = Random.Range(-1f, 1f);

        if (rand_num < -0.5f)
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        
        else if (rand_num < 0f)
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }

        else if (rand_num < 0.5f)
        {
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }

        else
        {
            velocity = new Vector3(Random.Range(-max_ball_speed, -min_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }
        
        RbBall.AddForce(velocity);
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("GoalA"))
        {
            AgentA.ScoredGoal();
            AgentB.OpponentScored();
            AgentA.AgentReset();
            AgentB.AgentReset();
        }

        if (collision.gameObject.CompareTag("GoalB"))
        {
            AgentB.ScoredGoal();
            AgentA.OpponentScored();
            AgentA.AgentReset();
            AgentB.AgentReset();
        }
    }
}
