using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NUnit.Framework;
using System.Reflection;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Sensors.Reflection;
using Unity.MLAgents.Policies;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.Utils.Tests;

namespace Unity.MLAgents.Tests
{
    [TestFixture]
    public class EditModeTestGeneration
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            var aca = Academy.Instance;
            Assert.AreNotEqual(null, aca);
            Assert.AreEqual(0, aca.EpisodeCount);
            Assert.AreEqual(0, aca.StepCount);
            Assert.AreEqual(0, aca.TotalStepCount);
        }

        [Test]
        public void TestAgent()
        {
            var agentGo = new GameObject("TestAgent");
            agentGo.AddComponent<TestAgent>();
            var agent = agentGo.GetComponent<TestAgent>();
            Assert.AreNotEqual(null, agent);
            Assert.AreEqual(0, agent.initializeAgentCalls);
        }
    }

    [TestFixture]
    public class EditModeTestInitialization
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            Assert.AreEqual(false, Academy.IsInitialized);
            var aca = Academy.Instance;
            Assert.AreEqual(true, Academy.IsInitialized);

            // Check that init is idempotent
            aca.LazyInitialize();
            aca.LazyInitialize();

            Assert.AreEqual(0, aca.EpisodeCount);
            Assert.AreEqual(0, aca.StepCount);
            Assert.AreEqual(0, aca.TotalStepCount);
            Assert.AreNotEqual(null, SideChannelManager.GetSideChannel<EnvironmentParametersChannel>());
            Assert.AreNotEqual(null, SideChannelManager.GetSideChannel<EngineConfigurationChannel>());
            Assert.AreNotEqual(null, SideChannelManager.GetSideChannel<StatsSideChannel>());

            // Check that Dispose is idempotent
            aca.Dispose();
            Assert.AreEqual(false, Academy.IsInitialized);
            aca.Dispose();
        }

        [Test]
        public void TestAcademyDispose()
        {
            var envParams1 = SideChannelManager.GetSideChannel<EnvironmentParametersChannel>();
            var engineParams1 = SideChannelManager.GetSideChannel<EngineConfigurationChannel>();
            var statsParams1 = SideChannelManager.GetSideChannel<StatsSideChannel>();
            Academy.Instance.Dispose();

            Academy.Instance.LazyInitialize();
            var envParams2 = SideChannelManager.GetSideChannel<EnvironmentParametersChannel>();
            var engineParams2 = SideChannelManager.GetSideChannel<EngineConfigurationChannel>();
            var statsParams2 = SideChannelManager.GetSideChannel<StatsSideChannel>();
            Academy.Instance.Dispose();

            Assert.AreNotEqual(envParams1, envParams2);
            Assert.AreNotEqual(engineParams1, engineParams2);
            Assert.AreNotEqual(statsParams1, statsParams2);
        }

        [Test]
        public void TestAgent()
        {
            var agentGo1 = new GameObject("TestAgent");
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var bp1 = agentGo1.GetComponent<BehaviorParameters>();
            bp1.ObservableAttributeHandling = ObservableAttributeOptions.ExcludeInherited;

            var agentGo2 = new GameObject("TestAgent");
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();

            Assert.AreEqual(0, agent1.agentOnEpisodeBeginCalls);
            Assert.AreEqual(0, agent2.agentOnEpisodeBeginCalls);
            Assert.AreEqual(0, agent1.initializeAgentCalls);
            Assert.AreEqual(0, agent2.initializeAgentCalls);
            Assert.AreEqual(0, agent1.agentActionCalls);
            Assert.AreEqual(0, agent2.agentActionCalls);


            agent2.LazyInitialize();
            agent1.LazyInitialize();

            // agent1 was not enabled when the academy started
            // The agents have been initialized
            Assert.AreEqual(0, agent1.agentOnEpisodeBeginCalls);
            Assert.AreEqual(0, agent2.agentOnEpisodeBeginCalls);
            Assert.AreEqual(1, agent1.initializeAgentCalls);
            Assert.AreEqual(1, agent2.initializeAgentCalls);
            Assert.AreEqual(0, agent1.agentActionCalls);
            Assert.AreEqual(0, agent2.agentActionCalls);

            // Make sure the Sensors were sorted
            Assert.AreEqual(agent1.sensors[0].GetName(), "observableFloat");
            Assert.AreEqual(agent1.sensors[1].GetName(), "testsensor1");
            Assert.AreEqual(agent1.sensors[2].GetName(), "testsensor2");

            // agent2 should only have two sensors (no observableFloat)
            Assert.AreEqual(agent2.sensors[0].GetName(), "testsensor1");
            Assert.AreEqual(agent2.sensors[1].GetName(), "testsensor2");
        }
    }

    [TestFixture]
    public class EditModeTestStep
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            var aca = Academy.Instance;

            var numberReset = 0;
            for (var i = 0; i < 10; i++)
            {
                Assert.AreEqual(numberReset, aca.EpisodeCount);
                Assert.AreEqual(i, aca.StepCount);

                // The reset happens at the beginning of the first step
                if (i == 0)
                {
                    numberReset += 1;
                }
                Academy.Instance.EnvironmentStep();
            }
        }

        [Test]
        public void TestAcademyAutostep()
        {
            var aca = Academy.Instance;
            Assert.IsTrue(aca.AutomaticSteppingEnabled);
            aca.AutomaticSteppingEnabled = false;
            Assert.IsFalse(aca.AutomaticSteppingEnabled);
            aca.AutomaticSteppingEnabled = true;
            Assert.IsTrue(aca.AutomaticSteppingEnabled);
        }

        [Test]
        public void TestAgent()
        {
            var agentGo1 = new GameObject("TestAgent");
            var bp1 = agentGo1.AddComponent<BehaviorParameters>();
            bp1.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            var bp2 = agentGo2.AddComponent<BehaviorParameters>();
            bp2.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();

            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.Awake();
            // agent1 will take an action at every step and request a decision every 2 steps
            // agent2 will request decisions only when RequestDecision is called

            agent1.LazyInitialize();

            var numberAgent1Episodes = 0;
            var numberAgent2Episodes = 0;
            var numberAgent2Initialization = 0;
            var requestDecision = 0;
            var requestAction = 0;
            for (var i = 0; i < 50; i++)
            {
                Assert.AreEqual(numberAgent1Episodes, agent1.agentOnEpisodeBeginCalls);
                Assert.AreEqual(numberAgent2Episodes, agent2.agentOnEpisodeBeginCalls);
                Assert.AreEqual(1, agent1.initializeAgentCalls);
                Assert.AreEqual(numberAgent2Initialization, agent2.initializeAgentCalls);
                Assert.AreEqual(i, agent1.agentActionCalls);
                Assert.AreEqual(requestAction, agent2.agentActionCalls);
                Assert.AreEqual((i + 1) / 2, agent1.collectObservationsCalls);
                Assert.AreEqual(requestDecision, agent2.collectObservationsCalls);
                // Agent 1 starts a new episode at the first step
                if (i == 0)
                {
                    numberAgent1Episodes += 1;
                }
                //Agent 2 is only initialized at step 2
                if (i == 2)
                {
                    // Since Agent2 is initialized after the Academy has stepped, its OnEpisodeBegin should be called now.
                    Assert.AreEqual(0, agent2.agentOnEpisodeBeginCalls);
                    agent2.LazyInitialize();
                    Assert.AreEqual(1, agent2.agentOnEpisodeBeginCalls);
                    numberAgent2Initialization += 1;
                    numberAgent2Episodes += 1;
                }

                // We are testing request decision and request actions when called
                // at different intervals
                if ((i % 3 == 0) && (i > 2))
                {
                    //Every 3 steps after agent 2 is initialized, request decision
                    requestDecision += 1;
                    requestAction += 1;
                    agent2.RequestDecision();
                }
                else if ((i % 5 == 0) && (i > 2))
                {
                    // Every 5 steps after agent 2 is initialized, request action
                    requestAction += 1;
                    agent2.RequestAction();
                }
                aca.EnvironmentStep();
            }
        }
    }

    [TestFixture]
    public class EditModeTestReset
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestAcademy()
        {
            var aca = Academy.Instance;

            var numberReset = 0;
            var stepsSinceReset = 0;
            for (var i = 0; i < 50; i++)
            {
                Assert.AreEqual(stepsSinceReset, aca.StepCount);
                Assert.AreEqual(numberReset, aca.EpisodeCount);
                Assert.AreEqual(i, aca.TotalStepCount);

                // Academy resets at the first step
                if (i == 0)
                {
                    numberReset += 1;
                }

                stepsSinceReset += 1;
                aca.EnvironmentStep();
            }
        }

        [Test]
        public void TestAgent()
        {
            var agentGo1 = new GameObject("TestAgent");
            var bp1 = agentGo1.AddComponent<BehaviorParameters>();
            bp1.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            var bp2 = agentGo2.AddComponent<BehaviorParameters>();
            bp2.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            agentGo2.AddComponent<TestAgent>();
            var agent2 = agentGo2.GetComponent<TestAgent>();

            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;

            agent2.LazyInitialize();

            var numberAgent1Episodes = 0;
            var numberAgent2Episodes = 0;
            var numberAcaReset = 0;
            var acaStepsSinceReset = 0;
            var agent2StepForEpisode = 0;
            for (var i = 0; i < 5000; i++)
            {
                Assert.AreEqual(acaStepsSinceReset, aca.StepCount);
                Assert.AreEqual(numberAcaReset, aca.EpisodeCount);

                Assert.AreEqual(i, aca.TotalStepCount);
                Assert.AreEqual(numberAgent2Episodes, agent2.agentOnEpisodeBeginCalls);
                Assert.AreEqual(agent2StepForEpisode, agent2.StepCount);

                // Agent 2 and academy reset at the first step
                if (i == 0)
                {
                    Assert.AreEqual(numberAgent2Episodes, agent2.agentOnEpisodeBeginCalls);
                    numberAcaReset += 1;
                    numberAgent2Episodes += 1;
                }

                //Agent 1 is only initialized at step 2
                if (i == 2)
                {
                    Assert.AreEqual(numberAgent1Episodes, agent1.agentOnEpisodeBeginCalls);
                    agent1.LazyInitialize();
                    numberAgent1Episodes += 1;
                    Assert.AreEqual(numberAgent1Episodes, agent1.agentOnEpisodeBeginCalls);
                }

                // Set agent 1 to done every 11 steps to test behavior
                if (i % 11 == 5)
                {
                    Assert.AreEqual(numberAgent1Episodes, agent1.agentOnEpisodeBeginCalls);
                    agent1.EndEpisode();
                    numberAgent1Episodes += 1;
                    Assert.AreEqual(numberAgent1Episodes, agent1.agentOnEpisodeBeginCalls);
                }

                // Ending the episode for agent 2 regularly
                if (i % 13 == 3)
                {
                    Assert.AreEqual(numberAgent2Episodes, agent2.agentOnEpisodeBeginCalls);
                    agent2.EndEpisode();
                    numberAgent2Episodes += 1;
                    agent2StepForEpisode = 0;
                    Assert.AreEqual(numberAgent2Episodes, agent2.agentOnEpisodeBeginCalls);
                }

                // Request a decision for agent 2 regularly
                if (i % 3 == 2)
                {
                    agent2.RequestDecision();
                }
                else if (i % 5 == 1)
                {
                    // Request an action without decision regularly
                    agent2.RequestAction();
                }

                acaStepsSinceReset += 1;
                agent2StepForEpisode += 1;
                aca.EnvironmentStep();
            }
        }
    }

    [TestFixture]
    public class EditModeTestMiscellaneous
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        [Test]
        public void TestCumulativeReward()
        {
            var agentGo1 = new GameObject("TestAgent");
            var bp1 = agentGo1.AddComponent<BehaviorParameters>();
            bp1.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            var agent1 = agentGo1.AddComponent<TestAgent>();
            var agentGo2 = new GameObject("TestAgent");
            var bp2 = agentGo2.AddComponent<BehaviorParameters>();
            bp2.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            var agent2 = agentGo2.AddComponent<TestAgent>();
            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 2;
            decisionRequester.Awake();


            agent1.MaxStep = 20;

            agent2.LazyInitialize();
            agent1.LazyInitialize();
            agent2.SetPolicy(new TestPolicy());

            var expectedAgent1ActionForEpisode = 0;

            for (var i = 0; i < 50; i++)
            {
                expectedAgent1ActionForEpisode += 1;
                if (expectedAgent1ActionForEpisode == agent1.MaxStep || i == 0)
                {
                    expectedAgent1ActionForEpisode = 0;
                }
                agent2.RequestAction();
                Assert.LessOrEqual(Mathf.Abs(expectedAgent1ActionForEpisode * 10.1f - agent1.GetCumulativeReward()), 0.05f);
                Assert.LessOrEqual(Mathf.Abs(i * 0.1f - agent2.GetCumulativeReward()), 0.05f);

                agent1.AddReward(10f);
                aca.EnvironmentStep();
            }
        }

        [Test]
        public void TestMaxStepsReset()
        {
            var agentGo1 = new GameObject("TestAgent");
            var bp1 = agentGo1.AddComponent<BehaviorParameters>();
            bp1.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 1;
            decisionRequester.Awake();

            const int maxStep = 6;
            agent1.MaxStep = maxStep;
            agent1.LazyInitialize();

            var expectedAgentStepCount = 0;
            var expectedEpisodes = 0;
            var expectedAgentAction = 0;
            var expectedAgentActionForEpisode = 0;
            var expectedCollectObsCalls = 0;
            var expectedCollectObsCallsForEpisode = 0;
            var expectedCompletedEpisodes = 0;
            var expectedSensorResetCalls = 0;

            for (var i = 0; i < 15; i++)
            {
                // Agent should observe and act on each Academy step
                expectedAgentAction += 1;
                expectedAgentActionForEpisode += 1;
                expectedCollectObsCalls += 1;
                expectedCollectObsCallsForEpisode += 1;
                expectedAgentStepCount += 1;

                // If the next step will put the agent at maxSteps, we expect it to reset
                if (agent1.StepCount == maxStep - 1 || (i == 0))
                {
                    expectedEpisodes += 1;
                }

                if (agent1.StepCount == maxStep - 1)
                {
                    expectedAgentActionForEpisode = 0;
                    expectedCollectObsCallsForEpisode = 0;
                    expectedAgentStepCount = 0;
                    expectedCompletedEpisodes++;
                    expectedSensorResetCalls++;
                    expectedCollectObsCalls += 1;
                }
                aca.EnvironmentStep();

                Assert.AreEqual(expectedAgentStepCount, agent1.StepCount);
                Assert.AreEqual(expectedEpisodes, agent1.agentOnEpisodeBeginCalls);
                Assert.AreEqual(expectedAgentAction, agent1.agentActionCalls);
                Assert.AreEqual(expectedAgentActionForEpisode, agent1.agentActionCallsForEpisode);
                Assert.AreEqual(expectedCollectObsCalls, agent1.collectObservationsCalls);
                Assert.AreEqual(expectedCollectObsCallsForEpisode, agent1.collectObservationsCallsForEpisode);
                Assert.AreEqual(expectedCompletedEpisodes, agent1.CompletedEpisodes);
                Assert.AreEqual(expectedSensorResetCalls, agent1.sensor1.numResetCalls);
            }
        }

        [Test]
        public void TestHeuristicPolicyStepsSensors()
        {
            // Make sure that Agents with HeuristicPolicies step their sensors each Academy step.
            var agentGo1 = new GameObject("TestAgent");
            var bp1 = agentGo1.AddComponent<BehaviorParameters>();
            bp1.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(1);
            agentGo1.AddComponent<TestAgent>();
            var agent1 = agentGo1.GetComponent<TestAgent>();
            var aca = Academy.Instance;

            var decisionRequester = agent1.gameObject.AddComponent<DecisionRequester>();
            decisionRequester.DecisionPeriod = 1;
            decisionRequester.Awake();

            agent1.LazyInitialize();
            Assert.AreEqual(agent1.GetPolicy().GetType(), typeof(HeuristicPolicy));

            var numSteps = 10;
            for (var i = 0; i < numSteps; i++)
            {
                aca.EnvironmentStep();
            }
            Assert.AreEqual(numSteps, agent1.heuristicCalls);
            Assert.AreEqual(numSteps, agent1.sensor1.numWriteCalls);
            Assert.AreEqual(numSteps, agent1.sensor2.numCompressedCalls);

            Assert.AreEqual(
                agent1.collectObservationsCallsForEpisode,
                agent1.GetStoredActionBuffers().ContinuousActions[0]
            );
        }

        [Test]
        public void TestNullList()
        {
            var nullList = new HeuristicPolicy.NullList();
            Assert.Throws<NotImplementedException>(() =>
            {
                _ = ((IEnumerable<float>)nullList).GetEnumerator();
            });

            Assert.Throws<NotImplementedException>(() =>
            {
                _ = ((IEnumerable)nullList).GetEnumerator();
            });

            Assert.Throws<NotImplementedException>(() =>
            {
                nullList.CopyTo(new[] { 0f }, 0);
            });

            nullList.Add(0);
            Assert.IsTrue(nullList.Count == 0);

            nullList.Clear();
            Assert.IsTrue(nullList.Count == 0);

            nullList.Add(0);
            Assert.IsFalse(nullList.Contains(0));
            Assert.IsFalse(nullList.Remove(0));
            Assert.IsFalse(nullList.IsReadOnly);
            Assert.IsTrue(nullList.IndexOf(0) == -1);
            nullList.Insert(0, 0);
            Assert.IsFalse(nullList.Count > 0);
            nullList.RemoveAt(0);
            Assert.IsTrue(nullList.Count == 0);
            Assert.IsTrue(Mathf.Approximately(0f, nullList[0]));
            Assert.IsTrue(Mathf.Approximately(0f, nullList[1]));
        }
    }

    [TestFixture]
    public class TestOnEnableOverride
    {
        public class OnEnableAgent : Agent
        {
            public bool callBase;

            protected override void OnEnable()
            {
                if (callBase)
                    base.OnEnable();
            }
        }

        static void _InnerAgentTestOnEnableOverride(bool callBase = false)
        {
            var go = new GameObject();
            var agent = go.AddComponent<OnEnableAgent>();
            agent.callBase = callBase;
            var onEnable = typeof(OnEnableAgent).GetMethod("OnEnable", BindingFlags.NonPublic | BindingFlags.Instance);
            var sendInfo = typeof(Agent).GetMethod("SendInfoToBrain", BindingFlags.NonPublic | BindingFlags.Instance);
            Assert.NotNull(onEnable);
            onEnable.Invoke(agent, null);
            Assert.NotNull(sendInfo);
            if (agent.callBase)
            {
                Assert.DoesNotThrow(() => sendInfo.Invoke(agent, null));
            }
            else
            {
                Assert.Throws<UnityAgentsException>(() =>
                {
                    try
                    {
                        sendInfo.Invoke(agent, null);
                    }
                    catch (TargetInvocationException e)
                    {
                        throw e.GetBaseException();
                    }
                });
            }
        }

        [Test]
        public void TestAgentCallBaseOnEnable()
        {
            _InnerAgentTestOnEnableOverride(true);
        }

        [Test]
        public void TestAgentDontCallBaseOnEnable()
        {
            _InnerAgentTestOnEnableOverride();
        }
    }

    [TestFixture]
    public class ObservableAttributeBehaviorTests
    {
        public class BaseObservableAgent : Agent
        {
            [Observable]
            public float BaseField;
        }

        public class DerivedObservableAgent : BaseObservableAgent
        {
            [Observable]
            public float DerivedField;
        }


        [Test]
        public void TestObservableAttributeBehaviorIgnore()
        {
            var variants = new[]
            {
                // No observables found
                (ObservableAttributeOptions.Ignore, 0),
                // Only DerivedField found
                (ObservableAttributeOptions.ExcludeInherited, 1),
                // DerivedField and BaseField found
                (ObservableAttributeOptions.ExamineAll, 2)
            };

            foreach (var (behavior, expectedNumSensors) in variants)
            {
                var go = new GameObject();
                var agent = go.AddComponent<DerivedObservableAgent>();
                var bp = go.GetComponent<BehaviorParameters>();
                bp.ObservableAttributeHandling = behavior;
                agent.LazyInitialize();
                int numAttributeSensors = 0;
                foreach (var sensor in agent.sensors)
                {
                    if (sensor.GetType() != typeof(VectorSensor))
                    {
                        numAttributeSensors++;
                    }
                }
                Assert.AreEqual(expectedNumSensors, numAttributeSensors);
            }
        }
    }

    [TestFixture]
    public class AgentRecursionTests
    {
        [SetUp]
        public void SetUp()
        {
            if (Academy.IsInitialized)
            {
                Academy.Instance.Dispose();
            }
        }

        class CollectObsEndEpisodeAgent : Agent
        {
            public override void CollectObservations(VectorSensor sensor)
            {
                // NEVER DO THIS IN REAL CODE!
                EndEpisode();
            }
        }

        class OnEpisodeBeginEndEpisodeAgent : Agent
        {
            public override void OnEpisodeBegin()
            {
                // NEVER DO THIS IN REAL CODE!
                EndEpisode();
            }
        }

        void TestRecursiveThrows<T>() where T : Agent
        {
            var gameObj = new GameObject();
            var agent = gameObj.AddComponent<T>();
            agent.LazyInitialize();
            agent.RequestDecision();

            Assert.Throws<UnityAgentsException>(() =>
            {
                Academy.Instance.EnvironmentStep();
            });
        }

        [Test]
        public void TestRecursiveCollectObsEndEpisodeThrows()
        {
            TestRecursiveThrows<CollectObsEndEpisodeAgent>();
        }

        [Test]
        public void TestRecursiveOnEpisodeBeginEndEpisodeThrows()
        {
            TestRecursiveThrows<OnEpisodeBeginEndEpisodeAgent>();
        }
    }
}
