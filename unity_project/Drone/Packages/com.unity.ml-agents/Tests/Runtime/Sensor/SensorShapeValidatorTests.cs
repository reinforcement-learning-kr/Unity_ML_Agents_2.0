using System.Collections.Generic;
using System.Text.RegularExpressions;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Unity.MLAgents.Sensors;

namespace Unity.MLAgents.Tests
{
    public class DummySensor : ISensor
    {
        string m_Name = "DummySensor";
        ObservationSpec m_ObservationSpec;

        public DummySensor(int dim1)
        {
            m_ObservationSpec = ObservationSpec.Vector(dim1);
        }

        public DummySensor(int dim1, int dim2)
        {
            m_ObservationSpec = ObservationSpec.VariableLength(dim1, dim2);
        }

        public DummySensor(int dim1, int dim2, int dim3)
        {
            m_ObservationSpec = ObservationSpec.Visual(dim1, dim2, dim3);
        }

        public string GetName()
        {
            return m_Name;
        }

        public ObservationSpec GetObservationSpec()
        {
            return m_ObservationSpec;
        }

        public byte[] GetCompressedObservation()
        {
            return null;
        }

        public int Write(ObservationWriter writer)
        {
            return this.ObservationSize();
        }

        public void Update() { }
        public void Reset() { }

        public CompressionSpec GetCompressionSpec()
        {
            return CompressionSpec.Default();
        }
    }

    public class SensorShapeValidatorTests
    {
        [Test]
        public void TestShapesAgree()
        {
            var validator = new SensorShapeValidator();
            var sensorList1 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5, 6) };
            validator.ValidateSensors(sensorList1);

            var sensorList2 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5, 6) };
            validator.ValidateSensors(sensorList2);
        }

        [Test]
        public void TestNumSensorMismatch()
        {
            var validator = new SensorShapeValidator();
            var sensorList1 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5, 6) };
            validator.ValidateSensors(sensorList1);

            var sensorList2 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), };
            LogAssert.Expect(LogType.Assert, "Number of Sensors must match. 3 != 2");
            validator.ValidateSensors(sensorList2);

            // Add the sensors in the other order
            validator = new SensorShapeValidator();
            validator.ValidateSensors(sensorList2);
            LogAssert.Expect(LogType.Assert, "Number of Sensors must match. 2 != 3");
            validator.ValidateSensors(sensorList1);
        }

        [Test]
        public void TestDimensionMismatch()
        {
            var validator = new SensorShapeValidator();
            var sensorList1 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5, 6) };
            validator.ValidateSensors(sensorList1);

            var sensorList2 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5) };
            LogAssert.Expect(LogType.Assert, new Regex("Sensor shapes must match.*"));
            validator.ValidateSensors(sensorList2);

            // Add the sensors in the other order
            validator = new SensorShapeValidator();
            validator.ValidateSensors(sensorList2);
            LogAssert.Expect(LogType.Assert, new Regex("Sensor shapes must match.*"));
            validator.ValidateSensors(sensorList1);
        }

        [Test]
        public void TestSizeMismatch()
        {
            var validator = new SensorShapeValidator();
            var sensorList1 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5, 6) };
            validator.ValidateSensors(sensorList1);

            var sensorList2 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5, 7) };
            LogAssert.Expect(LogType.Assert, new Regex("Sensor shapes must match.*"));
            validator.ValidateSensors(sensorList2);

            // Add the sensors in the other order
            validator = new SensorShapeValidator();
            validator.ValidateSensors(sensorList2);
            LogAssert.Expect(LogType.Assert, new Regex("Sensor shapes must match.*"));
            validator.ValidateSensors(sensorList1);
        }

        [Test]
        public void TestEverythingMismatch()
        {
            var validator = new SensorShapeValidator();
            var sensorList1 = new List<ISensor>() { new DummySensor(1), new DummySensor(2, 3), new DummySensor(4, 5, 6) };
            validator.ValidateSensors(sensorList1);

            var sensorList2 = new List<ISensor>() { new DummySensor(1), new DummySensor(9) };
            LogAssert.Expect(LogType.Assert, "Number of Sensors must match. 3 != 2");
            LogAssert.Expect(LogType.Assert, new Regex("Sensor shapes must match.*"));
            validator.ValidateSensors(sensorList2);

            // Add the sensors in the other order
            validator = new SensorShapeValidator();
            validator.ValidateSensors(sensorList2);
            LogAssert.Expect(LogType.Assert, "Number of Sensors must match. 2 != 3");
            LogAssert.Expect(LogType.Assert, new Regex("Sensor shapes must match.*"));
            validator.ValidateSensors(sensorList1);
        }
    }
}
