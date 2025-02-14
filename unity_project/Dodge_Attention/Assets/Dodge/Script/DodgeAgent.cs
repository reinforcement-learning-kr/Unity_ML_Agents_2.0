using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class DodgeAgent : Agent
{
    public Area area;
    Rigidbody RbAgent;
    float speed = 3f;
    Vector3 centerPos = Vector3.zero;

    public float DecisionWaitingTime = 0.05f;
    float m_currentTime = 0f;

    BufferSensorComponent m_BufferSensor;
    List<float> sensorDistList = new List<float>();

    float ray_length = 6f; 

    public override void Initialize()
    {
        RbAgent = gameObject.GetComponent<Rigidbody>();
        centerPos = gameObject.transform.position;

        Academy.Instance.AgentPreStep += WaitTimeInference;

        m_BufferSensor = gameObject.GetComponent<BufferSensorComponent>(); 
    }

    public override void OnEpisodeBegin()
    {
        area.ResetEnv();

        transform.localPosition = centerPos;
        RbAgent.velocity = Vector3.zero;
        RbAgent.angularVelocity = Vector3.zero; 
    }

    public void SetAgentSpeed(float speed_)
    {
        speed = speed_; 
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        RaycastHit hit;
        Ray ray;

        float angle;
        int raycount = 40;
        List<Vector3> debugRay = new List<Vector3>();

        for (int i = 0; i < raycount; i++)
        {
            angle = i * 2.0f * Mathf.PI / raycount;
            ray = new Ray(gameObject.transform.position, new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle)));

            if (Physics.Raycast(ray, out hit, ray_length))
            {
                sensorDistList.Clear();
                sensorDistList.Add(hit.distance * Mathf.Cos(angle) / ray_length);
                sensorDistList.Add(hit.distance * Mathf.Sin(angle) / ray_length);

                if (hit.collider.gameObject.name == "ArenaWalls")
                {
                    sensorDistList.Add(0f);
                    sensorDistList.Add(0f);
                }
                else
                {
                    Rigidbody rig = hit.collider.gameObject.GetComponent<Rigidbody>();
                    sensorDistList.Add(rig.velocity.x);
                    sensorDistList.Add(rig.velocity.z);
                }

                debugRay.Add(hit.point);
                m_BufferSensor.AppendObservation(sensorDistList.ToArray()); 
            }
        }

        sensor.AddObservation(RbAgent.velocity.x);
        sensor.AddObservation(RbAgent.velocity.z);

        for (int i = 0; i < debugRay.Count; i++)
        {
            Debug.DrawRay(gameObject.transform.position, debugRay[i] - gameObject.transform.position, Color.green); 
        }
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var action = actionBuffers.DiscreteActions[0];
        Vector3 force = Vector3.zero; 

        switch (action)
        {
            case 1: force = new Vector3(-1, 0, 0) * speed; break;
            case 2: force = new Vector3(0, 0, 1) * speed; break;
            case 3: force = new Vector3(0, 0, -1) * speed; break;
            case 4: force = new Vector3(1, 0, 0) * speed; break;
            default: force = new Vector3(0, 0, 0) * speed; break; 
        }

        RbAgent.AddForce(force, ForceMode.VelocityChange);

        Collider[] block = Physics.OverlapBox(gameObject.transform.position, Vector3.one * 0.5f); 

        if (block.Where(Col => Col.gameObject.CompareTag("ball")).ToArray().Length != 0)
        {
            SetReward(-1f);
            EndEpisode(); 
        }
        else
        {
            SetReward(0.1f); 
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        discreteActionsOut[0] = 0;

        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[0] = 2;
        }
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[0] = 3;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 4;
        }
    }

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
