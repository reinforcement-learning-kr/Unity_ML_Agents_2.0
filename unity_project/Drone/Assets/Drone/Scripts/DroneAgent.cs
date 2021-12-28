using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;
using PA_DronePack;

public class DroneAgent : Agent
{
    private PA_DroneController dcoScript;

    public DroneSetting area;
    public GameObject goal;

    float preDist;

    private Transform agentTrans;
    private Transform goalTrans;

    private Rigidbody agent_Rigidbody;


    public override void Initialize()
    {

        dcoScript = gameObject.GetComponent<PA_DroneController>();

        agentTrans = gameObject.transform;
        goalTrans = goal.transform;

        agent_Rigidbody = gameObject.GetComponent<Rigidbody>();

        Academy.Instance.AgentPreStep += WaitTimeInference;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // ∞≈∏Æ∫§≈Õ
        sensor.AddObservation(agentTrans.position - goalTrans.position);

        // º”µµ∫§≈Õ
        sensor.AddObservation(agent_Rigidbody.velocity);

        // ∞¢º”µµ ∫§≈Õ
        sensor.AddObservation(agent_Rigidbody.angularVelocity);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        AddReward(-0.01f);

        var actions = actionBuffers.ContinuousActions;

        float moveX = Mathf.Clamp(actions[0], -1, 1f);
        float moveY = Mathf.Clamp(actions[1], -1, 1f);
        float moveZ = Mathf.Clamp(actions[2], -1, 1f);

        dcoScript.DriveInput(moveX);
        dcoScript.StrafeInput(moveY);
        dcoScript.LiftInput(moveZ);

        float distance = Vector3.Magnitude(goalTrans.position - agentTrans.position);

        if(distance <= 0.5f)
        {
            SetReward(1f);
            EndEpisode();
        }
        else if(distance > 10f)
        {
            SetReward(-1f);
            EndEpisode();
        }
        else
        {
            float reward = preDist - distance;
            SetReward(reward);
            preDist = distance;
        }
    }

    public override void OnEpisodeBegin()
    {
        area.AreaSetting();

        preDist = Vector3.Magnitude(goalTrans.position - agentTrans.position);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;

        continuousActionsOut[0] = Input.GetAxis("Vertical");
        continuousActionsOut[1] = Input.GetAxis("Horizontal");
        continuousActionsOut[2] = Input.GetAxis("Mouse ScrollWheel");
    }

    public float DecisionWaitingTime = 5f;
    float m_currentTime = 0f;

    public void WaitTimeInference(int action)
    {
        if(Academy.Instance.IsCommunicatorOn)
        {
            RequestDecision();
        }
        else
        {
            if(m_currentTime >= DecisionWaitingTime)
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
