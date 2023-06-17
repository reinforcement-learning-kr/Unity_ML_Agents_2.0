using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators; 

public class TwoMissionAgent : Agent 
{
    public TwoMissionEnvController envController;
    public float runSpeed = 1f;
    private Rigidbody agentRb;
    public float[] goalsOneHotArr;

    VectorSensorComponent goalSensor;

    public override void Initialize()
    {
        envController.InitializeEnv();
        agentRb = GetComponent<Rigidbody>();
        goalSensor = this.GetComponent<VectorSensorComponent>();
        Academy.Instance.AgentPreStep += WaitTimeInference; 
    }

    public override void OnEpisodeBegin()
    {
        envController.ResetScene();
        goalsOneHotArr = envController.GoalsOneHotArr; 
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        goalSensor.GetSensor().AddObservation(goalsOneHotArr);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(this.transform.localRotation);
        sensor.AddObservation(agentRb.velocity);
        sensor.AddObservation(envController.ball.GetComponent<Rigidbody>().velocity); 
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        MoveAgent(actionBuffers.DiscreteActions);

        if (!envController.IsAgentOnGround() || !envController.IsBallOnGround())
        {
            SetReward(-1f);
            EndEpisode(); 
        }
    }

    public void MoveAgent(ActionSegment<int> act)
    {
        AddReward(-0.0005f);

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var action = act[0];

        switch (action)
        {
            case 0:
                dirToGo = transform.forward * 1f;
                break;
            case 1:
                dirToGo = transform.forward * -1f;
                break;
            case 2:
                rotateDir = transform.up * 1f;
                break;
            case 3:
                rotateDir = transform.up * -1f;
                break;
        }

        transform.Rotate(rotateDir, Time.fixedDeltaTime * 300f);
        agentRb.AddForce(dirToGo * runSpeed, ForceMode.VelocityChange);
    }

    private void OnTriggerEnter(Collider col)
    {
        if (col.gameObject.CompareTag("wallPassGoal"))
        {
            if (envController.CurrentGoal == GoalType.WallPass)
            {
                ReachedGoal(); 
            }
        }
    }

    public void ReachedGoal()
    {
        SetReward(1f);
        EndEpisode(); 
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 0;
        }
        else if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 1;
        }
        else if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 2;
        }
        else if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 3;
        }
    }

    float DecisionWaitingTime = 0.02f;
    float m_currentTime = 0f;

    public void WaitTimeInference(int action)
    {
        if (Academy.Instance.IsCommunicatorOn)
        {
            RequestDecision();
        }
        else
        {
            if (m_currentTime >= DecisionWaitingTime)
            {
                m_currentTime = 0f;
                RequestDecision();
            }
            else
            {
                m_currentTime += Time.fixedDeltaTime;
            }
        }
    }
}
