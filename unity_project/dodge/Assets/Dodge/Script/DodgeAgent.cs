using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

public class DodgeAgent : Agent
{
    public Area area;
    Rigidbody RbAgent;
    float speed = 30f;
    Vector3 centerPos = Vector3.zero;

    public float DecisionWaitingTime = 0.05f;
    float m_currentTime = 0f;

    public override void Initialize()
    {
		RbAgent = gameObject.GetComponent<Rigidbody>();
		centerPos = gameObject.transform.position;

		Academy.Instance.AgentPreStep += WaitTimeInference;
	}

    public override void OnEpisodeBegin()
    {
		transform.localPosition = centerPos;
		RbAgent.velocity = Vector3.zero;
		RbAgent.angularVelocity = Vector3.zero;

		area.ResetEnv();
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
		int rayCount = 40;
		List<Vector3> debugRay = new List<Vector3>();

		for(int i = 0; i < rayCount; i++)
        {
			angle = i * 2.0f * Mathf.PI / rayCount;
			ray = new Ray(gameObject.transform.position, new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle)));

			if(Physics.Raycast(ray, out hit))
            {
				sensor.AddObservation(hit.distance);

				if (hit.collider.gameObject.name == "ArenaWalls")
                {
					sensor.AddObservation(new Vector2());
                }
				else
                {
					Rigidbody rig = hit.collider.gameObject.GetComponent<Rigidbody>();
					var vel = new Vector2(rig.velocity.x, rig.velocity.z);
					sensor.AddObservation(vel);
                }

				debugRay.Add(hit.point);
			}
		}

		sensor.AddObservation(RbAgent.velocity.x);
		sensor.AddObservation(RbAgent.velocity.z);

		for (int i = 0; i < debugRay.Count; i++)
			Debug.DrawRay(gameObject.transform.position, debugRay[i] - gameObject.transform.position, Color.green);
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

		RbAgent.AddForce(force);

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

		if (Input.GetKey(KeyCode.D))
		{
			discreteActionsOut[0] = 4;
		}
		if (Input.GetKey(KeyCode.W))
		{
			discreteActionsOut[0] = 2;
		}
		if (Input.GetKey(KeyCode.A))
		{
			discreteActionsOut[0] = 1;
		}
		if (Input.GetKey(KeyCode.S))
		{
			discreteActionsOut[0] = 3;
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
