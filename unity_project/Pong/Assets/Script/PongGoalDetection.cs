using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PongGoalDetection : MonoBehaviour
{
    public PongAgent AgentA;
    public PongAgent AgentB;

    private Rigidbody RbBall;

    private Vector3 ResetPos;
    private Vector3 velocity;

    private Transform ballTrans = null;

    private float max_ball_speed = 10f;
    private float min_ball_speed = 5f;


    private void Start()
    {
        RbBall = gameObject.GetComponent<Rigidbody>();
        ballTrans = transform;
        ResetPos = ballTrans.position;
    }

    public void ResetPostion()
    {
        ballTrans.position = ResetPos;
        RbBall.velocity = Vector3.zero;
        RbBall.angularVelocity = Vector3.zero;
        ballTrans.rotation = Quaternion.identity;

        float rand_num = Random.Range(-1f, 1f);
    
        if(rand_num < -0.5f)     // 오른쪽 위로 이동
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(min_ball_speed, max_ball_speed));
        }
        else if (rand_num < 0f)     // 오른쪽 아로 이동
        {
            velocity = new Vector3(Random.Range(min_ball_speed, max_ball_speed), 0, Random.Range(-max_ball_speed, -min_ball_speed));
        }
        else if (rand_num < 0.5f)   // 왼쪽 위로 이동
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
        if(collision.gameObject.CompareTag("GoalA"))
        {
            AgentB.ScoredGoal();
            AgentA.OpponentScored();
            AgentA.AgentReset();
            AgentB.AgentReset();
        }

        if (collision.gameObject.CompareTag("GoalB"))
        {
            AgentA.ScoredGoal();
            AgentB.OpponentScored();
            AgentA.AgentReset();
            AgentB.AgentReset();
        }
    }
}
