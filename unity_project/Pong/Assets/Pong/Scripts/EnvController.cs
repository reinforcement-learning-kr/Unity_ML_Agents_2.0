using UnityEngine;

public class EnvController : MonoBehaviour
{
    public GameObject Ball;
    public PongAgent Agent_A;
    public PongAgent Agent_B;

    private Rigidbody RbBall;

    private Vector3 ResetPosBall;

    private float max_ball_speed = 8f;
    private float min_ball_speed = 5f;
    private float ball_x_vel_old = 0f;

    private int resetTimer;
    public int MaxEnvironmentSteps; 

    void Start()
    {
        ResetPosBall = Ball.transform.localPosition;
        RbBall = Ball.GetComponent<Rigidbody>();
        ResetScene(); 
    }

    private void FixedUpdate()
    {
        resetTimer += 1;
        if (resetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            Agent_A.EpisodeInterrupted();
            Agent_B.EpisodeInterrupted();
            ResetScene(); 
        }
        else
        {
            if (Mathf.Abs(RbBall.velocity.x) <= min_ball_speed)
            {
                if (RbBall.velocity.x > 0)
                {
                    RbBall.velocity = new Vector3(min_ball_speed, 0, RbBall.velocity.z);
                }
                else
                {
                    RbBall.velocity = new Vector3(-min_ball_speed, 0, RbBall.velocity.z);
                }
            }

            if (ball_x_vel_old < 0 && RbBall.velocity.x > 0)
            {
                Agent_A.AddReward(0.5f);
            }

            if (ball_x_vel_old > 0 && RbBall.velocity.x < 0)
            {
                Agent_B.AddReward(0.5f);
            }

            ball_x_vel_old = RbBall.velocity.x;

            if (Ball.transform.localPosition.x < -10.5f)
            {
                Agent_A.AddReward(-1f);
                Agent_B.AddReward(1f);
                Agent_A.EndEpisode();
                Agent_B.EndEpisode();
                ResetScene(); 
            }
            else if (Ball.transform.localPosition.x > 10.5f)
            {
                Agent_A.AddReward(1f);
                Agent_B.AddReward(-1f);
                Agent_A.EndEpisode();
                Agent_B.EndEpisode();
                ResetScene();
            }
        }
    }

    public void ResetScene()
    {
        resetTimer = 0;
        Ball.transform.localPosition = ResetPosBall;

        RbBall.velocity = Vector3.zero;
        RbBall.angularVelocity = Vector3.zero;
        Ball.transform.rotation = Quaternion.identity;

        ball_x_vel_old = 0f;

        float rand_num = Random.Range(-1f, 1f);
        Vector3 velocity;

        float x_rand_speed = Random.Range(min_ball_speed, max_ball_speed);
        float z_rand_speed = Random.Range(min_ball_speed, max_ball_speed);

        if (rand_num < -0.5f)
        {
            velocity = new Vector3(x_rand_speed, 0, z_rand_speed);
        }
        else if (rand_num < 0f)
        {
            velocity = new Vector3(x_rand_speed, 0, -z_rand_speed);
        }
        else if (rand_num < 0.5f)
        {
            velocity = new Vector3(-x_rand_speed, 0, z_rand_speed);
        }
        else
        {
            velocity = new Vector3(-x_rand_speed, 0, -z_rand_speed);
        }
        RbBall.AddForce(velocity); 
    }
}
