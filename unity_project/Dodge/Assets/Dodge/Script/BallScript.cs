using UnityEngine;

public class BallScript : MonoBehaviour
{
    GameObject Agent;
    float ball_speed;
    float ball_random;
    float board_radius;

    public void SetBall(GameObject Agent_, float ball_speed_, float ball_random_, float board_radius_)
    {
        Agent = Agent_;
        ball_speed = ball_speed_;
        ball_random = ball_random_;
        board_radius = board_radius_ - 0.55f;

        RandomBall(); 
    }

    public void RandomBall()
    {
        float theta = Random.Range(0, 2 * Mathf.PI);
        transform.localPosition = new Vector3(board_radius * Mathf.Cos(theta), 0.25f, board_radius * Mathf.Sin(theta));
        float randomAngle = Mathf.Atan2(Agent.transform.localPosition.z - transform.localPosition.z,
            Agent.transform.localPosition.x - transform.localPosition.x) + Random.Range(-ball_random, ball_random);
        float randomSpeed = ball_speed + Random.Range(-0.5f * ball_random, 0.5f * ball_random);

        Rigidbody rig = GetComponent<Rigidbody>();
        rig.velocity = new Vector3(randomSpeed * Mathf.Cos(randomAngle), 0, randomSpeed * Mathf.Sin(randomAngle)); 
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag("wall"))
        {
            RandomBall(); 
        }
    }
}
