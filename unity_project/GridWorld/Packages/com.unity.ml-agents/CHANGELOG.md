# Changelog

All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to
[Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [2.0.0-exp.1] - 2021-04-22
### Major Changes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- The minimum supported Unity version was updated to 2019.4. (#5166)
- Several breaking interface changes were made. See the
[Migration Guide](https://github.com/Unity-Technologies/ml-agents/blob/release_17_docs/docs/Migrating.md) for more
details.
- Some methods previously marked as `Obsolete` have been removed. If you were using these methods, you need to replace them with their supported counterpart.
- The interface for disabling discrete actions in `IDiscreteActionMask` has changed.
`WriteMask(int branch, IEnumerable<int> actionIndices)` was replaced with
`SetActionEnabled(int branch, int actionIndex, bool isEnabled)`. (#5060)
- IActuator now implements IHeuristicProvider. (#5110)
- `ISensor.GetObservationShape()` was removed, and `GetObservationSpec()` was added. The `ITypedSensor`
and `IDimensionPropertiesSensor` interfaces were removed. (#5127)
- `ISensor.GetCompressionType()` was removed, and `GetCompressionSpec()` was added. The `ISparseChannelSensor`
interface was removed. (#5164)
- The abstract method `SensorComponent.GetObservationShape()` was no longer being called, so it has been removed. (#5172)
- `SensorComponent.CreateSensor()` was replaced with `SensorComponent.CreateSensors()`, which returns an `ISensor[]`. (#5181)
- `Match3Sensor` was refactored to produce cell and special type observations separately, and `Match3SensorComponent` now
produces two `Match3Sensor`s (unless there are no special types). Previously trained models will have different observation
sizes and will need to be retrained. (#5181)
- The `AbstractBoard` class for integration with Match-3 games was changed to make it easier to support boards with
different sizes using the same model. For a summary of the interface changes, please see the Migration Guide. (##5189)
- Updated the Barracuda package to version `1.4.0-preview`(#5236)
- `GridSensor` has been refactored and moved to main package, with changes to both sensor interfaces and behaviors.
Exsisting GridSensor created by extension package will not work in newer version. Previously trained models will
need to be retrained. Please see the Migration Guide for more details. (#5256)
- Models trained with 1.x versions of ML-Agents will no longer work at inference if they were trained using recurrent neural networks (#5254)

### Minor Changes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- The `.onnx` models input names have changed. All input placeholders will now use the prefix `obs_` removing the distinction between visual and vector observations. In addition, the inputs and outputs of LSTM changed. Models created with this version will not be usable with previous versions of the package (#5080, #5236)
- The `.onnx` models discrete action output now contains the discrete actions values and not the logits. Models created with this version will not be usable with previous versions of the package (#5080)
- Added ML-Agents package settings. (#5027)
- Make com.unity.modules.unityanalytics an optional dependency. (#5109)
- Make com.unity.modules.physics and com.unity.modules.physics2d optional dependencies. (#5112)
- The default `InferenceDevice` is now `InferenceDevice.Default`, which is equivalent to `InferenceDevice.Burst`. If you
depend on the previous behavior, you can explicitly set the Agent's `InferenceDevice` to `InferenceDevice.CPU`. (#5175)
- Added support for `Goal Signal` as a type of observation. Trainers can now use HyperNetworks to process `Goal Signal`. Trainers with HyperNetworks are more effective at solving multiple tasks. (#5142, #5159, #5149)
- Modified the [GridWorld environment](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md#gridworld) to use the new `Goal Signal` feature. (#5193)
- `DecisionRequester.ShouldRequestDecision()` and `ShouldRequestAction()`methods were added. These are used to
determine whether `Agent.RequestDecision()` and `Agent.RequestAction()` are called (respectively). (#5223)
- `RaycastPerceptionSensor` now caches its raycast results; they can be accessed via `RayPerceptionSensor.RayPerceptionOutput`. (#5222)
- `ActionBuffers` are now reset to zero before being passed to `Agent.Heuristic()` and
`IHeuristicProvider.Heuristic()`. (#5227)
- `Agent` will now call `IDisposable.Dispose()` on all `ISensor`s that implement the `IDisposable` interface. (#5233)
- `CameraSensor`, `RenderTextureSensor`, and `Match3Sensor` will now reuse their `Texture2D`s, reducing the
amount of memory that needs to be allocated during runtime. (#5233)
- Optimzed `ObservationWriter.WriteTexture()` so that it doesn't call `Texture2D.GetPixels32()` for `RGB24` textures.
This results in much less memory being allocated during inference with `CameraSensor` and `RenderTextureSensor`. (#5233)
- The Match-3 integration utilities were moved from `com.unity.ml-agents.extensions` to `com.unity.ml-agents`. (#5259)

#### ml-agents / ml-agents-envs / gym-unity (Python)
- Some console output have been moved from `info` to `debug` and will not be printed by default. If you want all messages to be printed, you can run `mlagents-learn` with the `--debug` option or add the line `debug: true` at the top of the yaml config file. (#5211)
- When using a configuration YAML, it is required to define all behaviors found in a Unity
executable in the trainer configuration YAML, or specify `default_settings`. (#5210)
- The embedding size of attention layers used when a BufferSensor is in the scene has been changed. It is now fixed to 128 units. It might be impossible to resume training from a checkpoint of a previous version. (#5272)

### Bug Fixes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- Fixed a bug where sensors and actuators could get sorted inconsistently on different systems to different Culture
settings. Unfortunately, this may require retraining models if it changes the resulting order of the sensors
or actuators on your system. (#5194)
- Removed additional memory allocations that were occurring due to assert messages and iterating of DemonstrationRecorders. (#5246)
- Fixed a bug where agent trying to access unintialized fields when creating a new RayPerceptionSensorComponent on an agent. (#5261)
- Fixed a bug where the DemonstrationRecorder would throw a null reference exception if Num Steps To Record was > 0 and Record was turned off. (#5274)

#### ml-agents / ml-agents-envs / gym-unity (Python)
- Fixed a bug where --results-dir has no effect. (#5269)
- Fixed a bug where old `.pt` checkpoints were not deleted during training. (#5271)
- The `UnityToGymWrapper` initializer now accepts an optional `action_space_seed` seed. If this is specified, it will
be used to set the random seed on the resulting action space. (#5303)


## [1.9.1-preview] - 2021-04-13
### Major Changes
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The `--resume` flag now supports resuming experiments with additional reward providers or
 loading partial models if the network architecture has changed. See
 [here](https://github.com/Unity-Technologies/ml-agents/blob/release_16_docs/docs/Training-ML-Agents.md#loading-an-existing-model)
 for more details. (#5213)

### Bug Fixes
#### com.unity.ml-agents (C#)
- Fixed erroneous warnings when using the Demonstration Recorder. (#5216)

#### ml-agents / ml-agents-envs / gym-unity (Python)
- Fixed an issue which was causing increased variance when using LSTMs. Also fixed an issue with LSTM when used with POCA and `sequence_length` < `time_horizon`. (#5206)
- Fixed a bug where the SAC replay buffer would not be saved out at the end of a run, even if `save_replay_buffer` was enabled. (#5205)
- ELO now correctly resumes when loading from a checkpoint. (#5202)
- In the Python API, fixed `validate_action` to expect the right dimensions when `set_action_single_agent` is called. (#5208)
- In the `GymToUnityWrapper`, raise an appropriate warning if `step()` is called after an environment is done. (#5204)
- Fixed an issue where using one of the `gym` wrappers would override user-set log levels. (#5201)
## [1.9.0-preview] - 2021-03-17
### Major Changes
#### com.unity.ml-agents (C#)
- The `BufferSensor` and `BufferSensorComponent` have been added. They allow the Agent to observe variable number of entities. For an example, see the [Sorter environment](https://github.com/Unity-Technologies/ml-agents/blob/release_15_docs/docs/Learning-Environment-Examples.md#sorter). (#4909)
- The `SimpleMultiAgentGroup` class and `IMultiAgentGroup` interface have been added. These allow Agents to be given rewards and
  end episodes in groups. For examples, see the [Cooperative Push Block](https://github.com/Unity-Technologies/ml-agents/blob/release_15_docs/docs/Learning-Environment-Examples.md#cooperative-push-block), [Dungeon Escape](https://github.com/Unity-Technologies/ml-agents/blob/release_15_docs/docs/Learning-Environment-Examples.md#dungeon-escape) and [Soccer](https://github.com/Unity-Technologies/ml-agents/blob/release_15_docs/docs/Learning-Environment-Examples.md#soccer-twos) environments. (#4923)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The MA-POCA trainer has been added. This is a new trainer that enables Agents to learn how to work together in groups. Configure
  `poca` as the trainer in the configuration YAML after instantiating a `SimpleMultiAgentGroup` to use this feature. (#5005)

### Minor Changes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- Updated com.unity.barracuda to 1.3.2-preview. (#5084)
- Added 3D Ball to the `com.unity.ml-agents` samples. (#5077)
- Make com.unity.modules.unityanalytics an optional dependency. (#5109)

#### ml-agents / ml-agents-envs / gym-unity (Python)
- The `encoding_size` setting for RewardSignals has been deprecated. Please use `network_settings` instead. (#4982)
- Sensor names are now passed through to `ObservationSpec.name`. (#5036)

### Bug Fixes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- An issue that caused `GAIL` to fail for environments where agents can terminate episodes by self-sacrifice has been fixed. (#4971)
- Made the error message when observations of different shapes are sent to the trainer clearer. (#5030)
- An issue that prevented curriculums from incrementing with self-play has been fixed. (#5098)

## [1.8.1-preview] - 2021-03-08
### Minor Changes
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The `cattrs` version dependency was updated to allow `>=1.1.0` on Python 3.8 or higher. (#4821)

### Bug Fixes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- Fix an issue where queuing InputEvents overwrote data from previous events in the same frame. (#5034)

## [1.8.0-preview] - 2021-02-17
### Major Changes
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- TensorFlow trainers have been removed, please use the Torch trainers instead. (#4707)
- A plugin system for `mlagents-learn` has been added. You can now define custom
  `StatsWriter` implementations and register them to be called during training.
  More types of plugins will be added in the future. (#4788)

### Minor Changes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- The `ActionSpec` constructor is now public. Previously, it was not possible to create an
  ActionSpec with both continuous and discrete actions from code. (#4896)
- `StatAggregationMethod.Sum` can now be passed to `StatsRecorder.Add()`. This
  will result in the values being summed (instead of averaged) when written to
  TensorBoard. Thanks to @brccabral for the contribution! (#4816)
- The upper limit for the time scale (by setting the `--time-scale` paramater in mlagents-learn) was
  removed when training with a player. The Editor still requires it to be clamped to 100. (#4867)
- Added the IHeuristicProvider interface to allow IActuators as well as Agent implement the Heuristic function to generate actions.
  Updated the Basic example and the Match3 Example to use Actuators.
  Changed the namespace and file names of classes in com.unity.ml-agents.extensions. (#4849)
- Added `VectorSensor.AddObservation(IList<float>)`. `VectorSensor.AddObservation(IEnumerable<float>)`
  is deprecated. The `IList` version is recommended, as it does not generate any
  additional memory allocations. (#4887)
- Added `ObservationWriter.AddList()` and deprecated `ObservationWriter.AddRange()`.
  `AddList()` is recommended, as it does not generate any additional memory allocations. (#4887)
- The Barracuda dependency was upgraded to 1.3.0. (#4898)
- Added `ActuatorComponent.CreateActuators`, and deprecate `ActuatorComponent.CreateActuator`.  The
  default implementation will wrap `ActuatorComponent.CreateActuator` in an array and return that. (#4899)
- `InferenceDevice.Burst` was added, indicating that Agent's model will be run using Barracuda's Burst backend.
  This is the default for new Agents, but existing ones that use `InferenceDevice.CPU` should update to
  `InferenceDevice.Burst`. (#4925)
- Add an InputActuatorComponent to allow the generation of Agent action spaces from an InputActionAsset.
  Projects wanting to use this feature will need to add the
  [Input System Package](https://docs.unity3d.com/Packages/com.unity.inputsystem@1.1/manual/index.html)
  at version 1.1.0-preview.3 or later. (#4881)

#### ml-agents / ml-agents-envs / gym-unity (Python)
- Tensorboard now logs the Environment Reward as both a scalar and a histogram. (#4878)
- Added a `--torch-device` commandline option to `mlagents-learn`, which sets the default
  [`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device) used for training. (#4888)
- The `--cpu` commandline option had no effect and was removed. Use `--torch-device=cpu` to force CPU training. (#4888)
- The `mlagents_env` API has changed, `BehaviorSpec` now has a `observation_specs` property containing a list of `ObservationSpec`. For more information on `ObservationSpec` see [here](https://github.com/Unity-Technologies/ml-agents/blob/release_13_docs/docs/Python-API.md#behaviorspec). (#4763, #4825)

### Bug Fixes
#### com.unity.ml-agents (C#)
- Fix a compile warning about using an obsolete enum in `GrpcExtensions.cs`. (#4812)
- CameraSensor now logs an error if the GraphicsDevice is null. (#4880)
- Removed unnecessary memory allocations in `ActuatorManager.UpdateActionArray()` (#4877)
- Removed unnecessary memory allocations in `SensorShapeValidator.ValidateSensors()` (#4879)
- Removed unnecessary memory allocations in `SideChannelManager.GetSideChannelMessage()` (#4886)
- Removed several memory allocations that happened during inference. On a test scene, this
  reduced the amount of memory allocated by approximately 25%. (#4887)
- Removed several memory allocations that happened during inference with discrete actions. (#4922)
- Properly catch permission errors when writing timer files. (#4921)
- Unexpected exceptions during training initialization and shutdown are now logged. If you see
  "noisy" logs, please let us know! (#4930, #4935)

#### ml-agents / ml-agents-envs / gym-unity (Python)
- Fixed a bug that would cause an exception when `RunOptions` was deserialized via `pickle`. (#4842)
- Fixed a bug that can cause a crash if a behavior can appear during training in multi-environment training. (#4872)
- Fixed the computation of entropy for continuous actions. (#4869)
- Fixed a bug that would cause `UnityEnvironment` to wait the full timeout
  period and report a misleading error message if the executable crashed
  without closing the connection. It now periodically checks the process status
  while waiting for a connection, and raises a better error message if it crashes. (#4880)
- Passing a `-logfile` option in the `--env-args` option to `mlagents-learn` is
  no longer overwritten. (#4880)
- The `load_weights` function was being called unnecessarily often in the Ghost Trainer leading to training slowdowns. (#4934)


## [1.7.2-preview] - 2020-12-22
### Bug Fixes
#### com.unity.ml-agents (C#)
- Add analytics package dependency to the package manifest. (#4794)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- Fixed the docker build process. (#4791)


## [1.7.0-preview] - 2020-12-21
### Major Changes
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- PyTorch trainers now support training agents with both continuous and discrete action spaces. (#4702)
The `.onnx` models generated by the trainers of this release are incompatible with versions of Barracuda before `1.2.1-preview`. If you upgrade the trainers, you must upgrade the version of the Barracuda package as well (which can be done by upgrading the `com.unity.ml-agents` package).
### Minor Changes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- Agents with both continuous and discrete actions are now supported. You can specify
both continuous and discrete action sizes in Behavior Parameters. (#4702, #4718)
- In order to improve the developer experience for Unity ML-Agents Toolkit, we have added in-editor analytics.
Please refer to "Information that is passively collected by Unity" in the
[Unity Privacy Policy](https://unity3d.com/legal/privacy-policy). (#4677)
- The FoodCollector example environment now uses continuous actions for moving and
discrete actions for shooting. (#4746)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- `ActionSpec.validate_action()` now enforces that `UnityEnvironment.set_action_for_agent()` receives a 1D `np.array`. (#4691)

### Bug Fixes
#### com.unity.ml-agents (C#)
- Removed noisy warnings about API minor version mismatches in both the C# and python code. (#4688)
#### ml-agents / ml-agents-envs / gym-unity (Python)


## [1.6.0-preview] - 2020-11-18
### Major Changes
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)
 - PyTorch trainers are now the default. See the
 [installation docs](https://github.com/Unity-Technologies/ml-agents/blob/release_10_docs/docs/Installation.md) for
 more information on installing PyTorch. For the time being, TensorFlow is still available;
 you can use the TensorFlow backend by adding `--tensorflow` to the CLI, or
 adding `framework: tensorflow` in the configuration YAML. (#4517)

### Minor Changes
#### com.unity.ml-agents / com.unity.ml-agents.extensions (C#)
- The Barracuda dependency was upgraded to 1.1.2 (#4571)
- Utilities were added to `com.unity.ml-agents.extensions` to make it easier to
integrate with match-3 games. See the [readme](https://github.com/Unity-Technologies/ml-agents/blob/release_10_docs/com.unity.ml-agents.extensions/Documentation~/Match3.md)
for more details. (#4515)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The `action_probs` node is no longer listed as an output in TensorFlow models (#4613).

### Bug Fixes
#### com.unity.ml-agents (C#)
- `Agent.CollectObservations()` and `Agent.EndEpisode()` will now throw an exception
if they are called recursively (for example, if they call `Agent.EndEpisode()`).
Previously, this would result in an infinite loop and cause the editor to hang. (#4573)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- Fixed an issue where runs could not be resumed when using TensorFlow and Ghost Training. (#4593)
- Change the tensor type of step count from int32 to int64 to address the overflow issue when step
goes larger than 2^31. Previous Tensorflow checkpoints will become incompatible and cannot be loaded. (#4607)
- Remove extra period after "Training" in console log. (#4674)


## [1.5.0-preview] - 2020-10-14
### Major Changes
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)
 - Added the Random Network Distillation (RND) intrinsic reward signal to the Pytorch
 trainers. To use RND, add a `rnd` section to the `reward_signals` section of your
 yaml configuration file. [More information here](https://github.com/Unity-Technologies/ml-agents/blob/release_9_docs/docs/Training-Configuration-File.md#rnd-intrinsic-reward) (#4473)
### Minor Changes
#### com.unity.ml-agents (C#)
 - Stacking for compressed observations is now supported. An additional setting
 option `Observation Stacks` is added in editor to sensor components that support
 compressed observations. A new class `ISparseChannelSensor` with an
 additional method `GetCompressedChannelMapping()`is added to generate a mapping
 of the channels in compressed data to the actual channel after decompression,
 for the python side to decompress correctly. (#4476)
 - Added a new visual 3DBall environment. (#4513)
#### ml-agents / ml-agents-envs / gym-unity (Python)
 - The Communication API was changed to 1.2.0 to indicate support for stacked
 compressed observation. A new entry `compressed_channel_mapping` is added to the
 proto to handle decompression correctly. Newer versions of the package that wish to
 make use of this will also need a compatible version of the Python trainers. (#4476)
 - In the `VisualFoodCollector` scene, a vector flag representing the frozen state of
 the agent is added to the input observations in addition to the original first-person
 camera frame. The scene is able to train with the provided default config file. (#4511)
 - Added conversion to string for sampler classes to increase the verbosity of
 the curriculum lesson changes. The lesson updates would now output the sampler
 stats in addition to the lesson and parameter name to the console.  (#4484)
 - Localized documentation in Russian is added. Thanks to @SergeyMatrosov for
 the contribution. (#4529)
### Bug Fixes
#### com.unity.ml-agents (C#)
 - Fixed a bug where accessing the Academy outside of play mode would cause the
 Academy to get stepped multiple times when in play mode. (#4532)
#### ml-agents / ml-agents-envs / gym-unity (Python)


## [1.4.0-preview] - 2020-09-16
### Major Changes
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)

### Minor Changes
#### com.unity.ml-agents (C#)
- The `IActuator` interface and `ActuatorComponent` abstract class were added.
These are analogous to `ISensor` and `SensorComponent`, but for applying actions
for an Agent. They allow you to control the action space more programmatically
than defining the actions in the Agent's Behavior Parameters. See
[BasicActuatorComponent.cs](https://github.com/Unity-Technologies/ml-agents/blob/release_7_docs/Project/Assets/ML-Agents/Examples/Basic/Scripts/BasicActuatorComponent.cs)
 for an example of how to use them. (#4297, #4315)
- Update Barracuda to 1.1.1-preview (#4482)
- Enabled C# formatting using `dotnet-format`. (#4362)
- GridSensor was added to the `com.unity.ml-agents.extensions` package. Thank you
to Jaden Travnik from Eidos Montreal for the contribution! (#4399)
- Added `Agent.EpisodeInterrupted()`, which can be used to reset the agent when
it has reached a user-determined maximum number of steps. This behaves similarly
to `Agent.EndEpsiode()` but has a slightly different effect on training (#4453).
#### ml-agents / ml-agents-envs / gym-unity (Python)
- Experimental PyTorch support has been added. Use `--torch` when running `mlagents-learn`, or add
`framework: pytorch` to your trainer configuration (under the behavior name) to enable it.
Note that PyTorch 1.6.0 or greater should be installed to use this feature; see
[the PyTorch website](https://pytorch.org/) for installation instructions and
[the relevant ML-Agents docs](https://github.com/Unity-Technologies/ml-agents/blob/release_7_docs/docs/Training-ML-Agents.md#using-pytorch-experimental) for usage. (#4335)
- The minimum supported version of TensorFlow was increased to 1.14.0. (#4411)
- Compressed visual observations with >3 channels are now supported. In
`ISensor.GetCompressedObservation()`, this can be done by writing 3 channels at a
time to a PNG and concatenating the resulting bytes. (#4399)
- The Communication API was changed to 1.1.0 to indicate support for concatenated PNGs
(see above). Newer versions of the package that wish to make use of this will also need
a compatible version of the trainer. (#4462)
- A CNN (`vis_encode_type: match3`) for smaller grids, e.g. board games, has been added.
(#4434)
- You can now again specify a default configuration for your behaviors. Specify `default_settings` in
your trainer configuration to do so. (#4448)
- Improved the executable detection logic for environments on Windows. (#4485)

### Bug Fixes
#### com.unity.ml-agents (C#)
- Previously, `com.unity.ml-agents` was not declaring built-in packages as
dependencies in its package.json. The relevant dependencies are now listed. (#4384)
- Agents no longer try to send observations when they become disabled if the
Academy has been shut down. (#4489)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- Fixed the sample code in the custom SideChannel example. (#4466)
- A bug in the observation normalizer that would cause rewards to decrease
when using `--resume` was fixed. (#4463)
- Fixed a bug in exporting Pytorch models when using multiple discrete actions. (#4491)

## [1.3.0-preview] - 2020-08-12

### Major Changes
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The minimum supported Python version for ml-agents-envs was changed to 3.6.1. (#4244)
- The interaction between EnvManager and TrainerController was changed; EnvManager.advance() was split into to stages,
and TrainerController now uses the results from the first stage to handle new behavior names. This change speeds up
Python training by approximately 5-10%. (#4259)

### Minor Changes
#### com.unity.ml-agents (C#)
- StatsSideChannel now stores multiple values per key. This means that multiple
calls to `StatsRecorder.Add()` with the same key in the same step will no
longer overwrite each other. (#4236)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The versions of `numpy` supported by ml-agents-envs were changed to disallow 1.19.0 or later. This was done to reflect
a similar change in TensorFlow's requirements. (#4274)
- Model checkpoints are now also saved as .nn files during training. (#4127)
- Model checkpoint info is saved in TrainingStatus.json after training is concluded (#4127)
- CSV statistics writer was removed (#4300).

### Bug Fixes
#### com.unity.ml-agents (C#)
- Academy.EnvironmentStep() will now throw an exception if it is called
recursively (for example, by an Agent's CollectObservations method).
Previously, this would result in an infinite loop and cause the editor to hang.
(#4226)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The algorithm used to normalize observations was introducing NaNs if the initial observations were too large
due to incorrect initialization. The initialization was fixed and is now the observation means from the
first trajectory processed. (#4299)

## [1.2.0-preview] - 2020-07-15

### Major Changes
#### ml-agents / ml-agents-envs / gym-unity (Python)
- The Parameter Randomization feature has been refactored to enable sampling of new parameters per episode to improve robustness. The
  `resampling-interval` parameter has been removed and the config structure updated. More information [here](https://github.com/Unity-Technologies/ml-agents/blob/release_5_docs/docs/Training-ML-Agents.md). (#4065)
- The Parameter Randomization feature has been merged with the Curriculum feature. It is now possible to specify a sampler
in the lesson of a Curriculum. Curriculum has been refactored and is now specified at the level of the parameter, not the
behavior. More information
[here](https://github.com/Unity-Technologies/ml-agents/blob/release_5_docs/docs/Training-ML-Agents.md).(#4160)

### Minor Changes
#### com.unity.ml-agents (C#)
- `SideChannelsManager` was renamed to `SideChannelManager`. The old name is still supported, but deprecated. (#4137)
- `RayPerceptionSensor.Perceive()` now additionally store the GameObject that was hit by the ray. (#4111)
- The Barracuda dependency was upgraded to 1.0.1 (#4188)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- Added new Google Colab notebooks to show how to use `UnityEnvironment'. (#4117)

### Bug Fixes
#### com.unity.ml-agents (C#)
- Fixed an issue where RayPerceptionSensor would raise an exception when the
list of tags was empty, or a tag in the list was invalid (unknown, null, or
empty string). (#4155)

#### ml-agents / ml-agents-envs / gym-unity (Python)
- Fixed an error when setting `initialize_from` in the trainer confiiguration YAML to
`null`. (#4175)
- Fixed issue with FoodCollector, Soccer, and WallJump when playing with keyboard. (#4147, #4174)
- Fixed a crash in StatsReporter when using threaded trainers with very frequent summary writes
(#4201)
- `mlagents-learn` will now raise an error immediately if `--num-envs` is greater than 1 without setting the `--env`
argument. (#4203)

## [1.1.0-preview] - 2020-06-10
### Major Changes
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- Added new Walker environments. Improved ragdoll stability/performance. (#4037)
- `max_step` in the `TerminalStep` and `TerminalSteps` objects was renamed `interrupted`.
- `beta` and `epsilon` in `PPO` are no longer decayed by default but follow the same schedule as learning rate. (#3940)
- `get_behavior_names()` and `get_behavior_spec()` on UnityEnvironment were replaced by the `behavior_specs` property. (#3946)
- The first version of the Unity Environment Registry (Experimental) has been released. More information [here](https://github.com/Unity-Technologies/ml-agents/blob/release_5_docs/docs/Unity-Environment-Registry.md)(#3967)
- `use_visual` and `allow_multiple_visual_obs` in the `UnityToGymWrapper` constructor
were replaced by `allow_multiple_obs` which allows one or more visual observations and
vector observations to be used simultaneously. (#3981) Thank you @shakenes !
- Curriculum and Parameter Randomization configurations have been merged
  into the main training configuration file. Note that this means training
  configuration files are now environment-specific. (#3791)
- The format for trainer configuration has changed, and the "default" behavior has been deprecated.
  See the [Migration Guide](https://github.com/Unity-Technologies/ml-agents/blob/release_5_docs/docs/Migrating.md) for more details. (#3936)
- Training artifacts (trained models, summaries) are now found in the `results/`
  directory. (#3829)
- When using Curriculum, the current lesson will resume if training is quit and resumed. As such,
  the `--lesson` CLI option has been removed. (#4025)
### Minor Changes
#### com.unity.ml-agents (C#)
- `ObservableAttribute` was added. Adding the attribute to fields or properties on an Agent will allow it to generate
  observations via reflection. (#3925, #4006)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- Unity Player logs are now written out to the results directory. (#3877)
- Run configuration YAML files are written out to the results directory at the end of the run. (#3815)
- The `--save-freq` CLI option has been removed, and replaced by a `checkpoint_interval` option in the trainer configuration YAML. (#4034)
- When trying to load/resume from a checkpoint created with an earlier verison of ML-Agents,
  a warning will be thrown. (#4035)
### Bug Fixes
- Fixed an issue where SAC would perform too many model updates when resuming from a
  checkpoint, and too few when using `buffer_init_steps`. (#4038)
- Fixed a bug in the onnx export that would cause constants needed for inference to not be visible to some versions of
  the Barracuda importer. (#4073)
#### com.unity.ml-agents (C#)
#### ml-agents / ml-agents-envs / gym-unity (Python)


## [1.0.2-preview] - 2020-05-20
### Bug Fixes
#### com.unity.ml-agents (C#)
- Fix missing .meta file


## [1.0.1-preview] - 2020-05-19
### Bug Fixes
#### com.unity.ml-agents (C#)
- A bug that would cause the editor to go into a loop when a prefab was selected was fixed. (#3949)
- BrainParameters.ToProto() no longer throws an exception if none of the fields have been set. (#3930)
- The Barracuda dependency was upgraded to 0.7.1-preview. (#3977)
#### ml-agents / ml-agents-envs / gym-unity (Python)
- An issue was fixed where using `--initialize-from` would resume from the past step count. (#3962)
- The gym wrapper error for the wrong number of agents now fires more consistently, and more details
  were added to the error message when the input dimension is wrong. (#3963)


## [1.0.0-preview] - 2020-04-30
### Major Changes
#### com.unity.ml-agents (C#)

- The `MLAgents` C# namespace was renamed to `Unity.MLAgents`, and other nested
  namespaces were similarly renamed. (#3843)
- The offset logic was removed from DecisionRequester. (#3716)
- The signature of `Agent.Heuristic()` was changed to take a float array as a
  parameter, instead of returning the array. This was done to prevent a common
  source of error where users would return arrays of the wrong size. (#3765)
- The communication API version has been bumped up to 1.0.0 and will use
  [Semantic Versioning](https://semver.org/) to do compatibility checks for
  communication between Unity and the Python process. (#3760)
- The obsolete `Agent` methods `GiveModel`, `Done`, `InitializeAgent`,
  `AgentAction` and `AgentReset` have been removed. (#3770)
- The SideChannel API has changed:
  - Introduced the `SideChannelManager` to register, unregister and access side
    channels. (#3807)
  - `Academy.FloatProperties` was replaced by `Academy.EnvironmentParameters`.
    See the [Migration Guide](https://github.com/Unity-Technologies/ml-agents/blob/release_1_docs/docs/Migrating.md)
    for more details on upgrading. (#3807)
  - `SideChannel.OnMessageReceived` is now a protected method (was public)
  - SideChannel IncomingMessages methods now take an optional default argument,
    which is used when trying to read more data than the message contains. (#3751)
  - Added a feature to allow sending stats from C# environments to TensorBoard
    (and other python StatsWriters). To do this from your code, use
    `Academy.Instance.StatsRecorder.Add(key, value)`. (#3660)
- `CameraSensorComponent.m_Grayscale` and
  `RenderTextureSensorComponent.m_Grayscale` were changed from `public` to
  `private`. These can still be accessed via their corresponding properties.
  (#3808)
- Public fields and properties on several classes were renamed to follow Unity's
  C# style conventions. All public fields and properties now use "PascalCase"
  instead of "camelCase"; for example, `Agent.maxStep` was renamed to
  `Agent.MaxStep`. For a full list of changes, see the pull request. (#3828)
- `WriteAdapter` was renamed to `ObservationWriter`. If you have a custom
  `ISensor` implementation, you will need to change the signature of its
  `Write()` method. (#3834)
- The Barracuda dependency was upgraded to 0.7.0-preview (which has breaking
  namespace and assembly name changes). (#3875)

#### ml-agents / ml-agents-envs / gym-unity (Python)

- The `--load` and `--train` command-line flags have been deprecated. Training
  now happens by default, and use `--resume` to resume training instead of
  `--load`. (#3705)
- The Jupyter notebooks have been removed from the repository. (#3704)
- The multi-agent gym option was removed from the gym wrapper. For multi-agent
  scenarios, use the [Low Level Python API](https://github.com/Unity-Technologies/ml-agents/blob/release_1_docs/docs/Python-API.md). (#3681)
- The low level Python API has changed. You can look at the document
  [Low Level Python API](https://github.com/Unity-Technologies/ml-agents/blob/release_1_docs/docs/Python-API.md)
  documentation for more information. If you use `mlagents-learn` for training, this should be a
  transparent change. (#3681)
- Added ability to start training (initialize model weights) from a previous run
  ID. (#3710)
- The GhostTrainer has been extended to support asymmetric games and the
  asymmetric example environment Strikers Vs. Goalie has been added. (#3653)
- The `UnityEnv` class from the `gym-unity` package was renamed
  `UnityToGymWrapper` and no longer creates the `UnityEnvironment`. Instead, the
  `UnityEnvironment` must be passed as input to the constructor of
  `UnityToGymWrapper` (#3812)

### Minor Changes

#### com.unity.ml-agents (C#)

- Added new 3-joint Worm ragdoll environment. (#3798)
- `StackingSensor` was changed from `internal` visibility to `public`. (#3701)
- The internal event `Academy.AgentSetStatus` was renamed to
  `Academy.AgentPreStep` and made public. (#3716)
- Academy.InferenceSeed property was added. This is used to initialize the
  random number generator in ModelRunner, and is incremented for each
  ModelRunner. (#3823)
- `Agent.GetObservations()` was added, which returns a read-only view of the
  observations added in `CollectObservations()`. (#3825)
- `UnityRLCapabilities` was added to help inform users when RL features are
  mismatched between C# and Python packages. (#3831)

#### ml-agents / ml-agents-envs / gym-unity (Python)

- Format of console output has changed slightly and now matches the name of the
  model/summary directory. (#3630, #3616)
- Renamed 'Generalization' feature to 'Environment Parameter Randomization'.
  (#3646)
- Timer files now contain a dictionary of metadata, including things like the
  package version numbers. (#3758)
- The way that UnityEnvironment decides the port was changed. If no port is
  specified, the behavior will depend on the `file_name` parameter. If it is
  `None`, 5004 (the editor port) will be used; otherwise 5005 (the base
  environment port) will be used. (#3673)
- Running `mlagents-learn` with the same `--run-id` twice will no longer
  overwrite the existing files. (#3705)
- Model updates can now happen asynchronously with environment steps for better
  performance. (#3690)
- `num_updates` and `train_interval` for SAC were replaced with
  `steps_per_update`. (#3690)
- The maximum compatible version of tensorflow was changed to allow tensorflow
  2.1 and 2.2. This will allow use with python 3.8 using tensorflow 2.2.0rc3.
  (#3830)
- `mlagents-learn` will no longer set the width and height of the executable
  window to 84x84 when no width nor height arguments are given. (#3867)

### Bug Fixes

#### com.unity.ml-agents (C#)

- Fixed a display bug when viewing Demonstration files in the inspector. The
  shapes of the observations in the file now display correctly. (#3771)

#### ml-agents / ml-agents-envs / gym-unity (Python)

- Fixed an issue where exceptions from environments provided a return code of 0.
  (#3680)
- Self-Play team changes will now trigger a full environment reset. This
  prevents trajectories in progress during a team change from getting into the
  buffer. (#3870)

## [0.15.1-preview] - 2020-03-30

### Bug Fixes

- Raise the wall in CrawlerStatic scene to prevent Agent from falling off.
  (#3650)
- Fixed an issue where specifying `vis_encode_type` was required only for SAC.
  (#3677)
- Fixed the reported entropy values for continuous actions (#3684)
- Fixed an issue where switching models using `SetModel()` during training would
  use an excessive amount of memory. (#3664)
- Environment subprocesses now close immediately on timeout or wrong API
  version. (#3679)
- Fixed an issue in the gym wrapper that would raise an exception if an Agent
  called EndEpisode multiple times in the same step. (#3700)
- Fixed an issue where logging output was not visible; logging levels are now
  set consistently. (#3703)

## [0.15.0-preview] - 2020-03-18

### Major Changes

- `Agent.CollectObservations` now takes a VectorSensor argument. (#3352, #3389)
- Added `Agent.CollectDiscreteActionMasks` virtual method with a
  `DiscreteActionMasker` argument to specify which discrete actions are
  unavailable to the Agent. (#3525)
- Beta support for ONNX export was added. If the `tf2onnx` python package is
  installed, models will be saved to `.onnx` as well as `.nn` format. Note that
  Barracuda 0.6.0 or later is required to import the `.onnx` files properly
- Multi-GPU training and the `--multi-gpu` option has been removed temporarily.
  (#3345)
- All Sensor related code has been moved to the namespace `MLAgents.Sensors`.
- All SideChannel related code has been moved to the namespace
  `MLAgents.SideChannels`.
- `BrainParameters` and `SpaceType` have been removed from the public API
- `BehaviorParameters` have been removed from the public API.
- The following methods in the `Agent` class have been deprecated and will be
  removed in a later release:
  - `InitializeAgent()` was renamed to `Initialize()`
  - `AgentAction()` was renamed to `OnActionReceived()`
  - `AgentReset()` was renamed to `OnEpisodeBegin()`
  - `Done()` was renamed to `EndEpisode()`
  - `GiveModel()` was renamed to `SetModel()`

### Minor Changes

- Monitor.cs was moved to Examples. (#3372)
- Automatic stepping for Academy is now controlled from the
  AutomaticSteppingEnabled property. (#3376)
- The GetEpisodeCount, GetStepCount, GetTotalStepCount and methods of Academy
  were changed to EpisodeCount, StepCount, TotalStepCount properties
  respectively. (#3376)
- Several classes were changed from public to internal visibility. (#3390)
- Academy.RegisterSideChannel and UnregisterSideChannel methods were added.
  (#3391)
- A tutorial on adding custom SideChannels was added (#3391)
- The stepping logic for the Agent and the Academy has been simplified (#3448)
- Update Barracuda to 0.6.1-preview

* The interface for `RayPerceptionSensor.PerceiveStatic()` was changed to take
  an input class and write to an output class, and the method was renamed to
  `Perceive()`.

- The checkpoint file suffix was changed from `.cptk` to `.ckpt` (#3470)
- The command-line argument used to determine the port that an environment will
  listen on was changed from `--port` to `--mlagents-port`.
- `DemonstrationRecorder` can now record observations outside of the editor.
- `DemonstrationRecorder` now has an optional path for the demonstrations. This
  will default to `Application.dataPath` if not set.
- `DemonstrationStore` was changed to accept a `Stream` for its constructor, and
  was renamed to `DemonstrationWriter`
- The method `GetStepCount()` on the Agent class has been replaced with the
  property getter `StepCount`
- `RayPerceptionSensorComponent` and related classes now display the debug
  gizmos whenever the Agent is selected (not just Play mode).
- Most fields on `RayPerceptionSensorComponent` can now be changed while the
  editor is in Play mode. The exceptions to this are fields that affect the
  number of observations.
- Most fields on `CameraSensorComponent` and `RenderTextureSensorComponent` were
  changed to private and replaced by properties with the same name.
- Unused static methods from the `Utilities` class (ShiftLeft, ReplaceRange,
  AddRangeNoAlloc, and GetSensorFloatObservationSize) were removed.
- The `Agent` class is no longer abstract.
- SensorBase was moved out of the package and into the Examples directory.
- `AgentInfo.actionMasks` has been renamed to `AgentInfo.discreteActionMasks`.
- `DecisionRequester` has been made internal (you can still use the
  DecisionRequesterComponent from the inspector). `RepeatAction` was renamed
  `TakeActionsBetweenDecisions` for clarity. (#3555)
- The `IFloatProperties` interface has been removed.
- Fix #3579.
- Improved inference performance for models with multiple action branches.
  (#3598)
- Fixed an issue when using GAIL with less than `batch_size` number of
  demonstrations. (#3591)
- The interfaces to the `SideChannel` classes (on C# and python) have changed to
  use new `IncomingMessage` and `OutgoingMessage` classes. These should make
  reading and writing data to the channel easier. (#3596)
- Updated the ExpertPyramid.demo example demonstration file (#3613)
- Updated project version for example environments to 2018.4.18f1. (#3618)
- Changed the Product Name in the example environments to remove spaces, so that
  the default build executable file doesn't contain spaces. (#3612)

## [0.14.1-preview] - 2020-02-25

### Bug Fixes

- Fixed an issue which caused self-play training sessions to consume a lot of
  memory. (#3451)
- Fixed an IndexError when using GAIL or behavioral cloning with demonstrations
  recorded with 0.14.0 or later (#3464)
- Updated the `gail_config.yaml` to work with per-Agent steps (#3475)
- Fixed demonstration recording of experiences when the Agent is done. (#3463)
- Fixed a bug with the rewards of multiple Agents in the gym interface (#3471,
  #3496)

## [0.14.0-preview] - 2020-02-13

### Major Changes

- A new self-play mechanism for training agents in adversarial scenarios was
  added (#3194)
- Tennis and Soccer environments were refactored to enable training with
  self-play (#3194, #3331)
- UnitySDK folder was split into a Unity Package (com.unity.ml-agents) and our
  examples were moved to the Project folder (#3267)
- Academy is now a singleton and is no longer abstract (#3210, #3184)
- In order to reduce the size of the API, several classes and methods were
  marked as internal or private. Some public fields on the Agent were trimmed
  (#3342, #3353, #3269)
- Decision Period and on-demand decision checkboxes were removed from the Agent.
  on-demand decision is now the default (#3243)
- Calling Done() on the Agent will reset it immediately and call the AgentReset
  virtual method (#3291, #3242)
- The "Reset on Done" setting in AgentParameters was removed; this is now always
  true. AgentOnDone virtual method on the Agent was removed (#3311, #3222)
- Trainer steps are now counted per-Agent, not per-environment as in previous
  versions. For instance, if you have 10 Agents in the scene, 20 environment
  steps now correspond to 200 steps as printed in the terminal and in
  Tensorboard (#3113)

### Minor Changes

- Barracuda was updated to 0.5.0-preview (#3329)
- --num-runs option was removed from mlagents-learn (#3155)
- Curriculum config files are now YAML formatted and all curricula for a
  training run are combined into a single file (#3186)
- ML-Agents components, such as BehaviorParameters and various Sensor
  implementations, now appear in the Components menu (#3231)
- Exceptions are now raised in Unity (in debug mode only) if NaN observations or
  rewards are passed (#3221)
- RayPerception MonoBehavior, which was previously deprecated, was removed
  (#3304)
- Uncompressed visual (i.e. 3d float arrays) observations are now supported.
  CameraSensorComponent and RenderTextureSensor now have an option to write
  uncompressed observations (#3148)
- Agent’s handling of observations during training was improved so that an extra
  copy of the observations is no longer maintained (#3229)
- Error message for missing trainer config files was improved to include the
  absolute path (#3230)
- Support for 2017.4 LTS was dropped (#3121, #3168)
- Some documentation improvements were made (#3296, #3292, #3295, #3281)

### Bug Fixes

- Numpy warning when stats don’t exist (#3251)
- A bug that caused RayPerceptionSensor to behave inconsistently with transforms
  that have non-1 scale was fixed (#3321)
- Some small bugfixes to tensorflow_to_barracuda.py were backported from the
  barracuda release (#3341)
- Base port in the jupyter notebook example was updated to use the same port
  that the editor uses (#3283)

## [0.13.0-preview] - 2020-01-24

### This is the first release of _Unity Package ML-Agents_.

_Short description of this release_
