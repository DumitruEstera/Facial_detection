import React from 'react';

const PersonRegistration = ({ onRegister }) => {
  const [formData, setFormData] = React.useState({
    name: '',
    employee_id: '',
    department: '',
    authorized_zones: []
  });
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [message, setMessage] = React.useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setMessage('');

    try {
      const result = await onRegister(formData);
      setMessage({ type: 'success', text: result.message || 'Person registered successfully!' });
      setFormData({ name: '', employee_id: '', department: '', authorized_zones: [] });
    } catch (error) {
      setMessage({ type: 'error', text: 'Error registering person: ' + error.message });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  return (
    <div className="registration-form">
      <h2>👤 Register New Person</h2>
      
      {message && (
        <div className={`message ${message.type}`}>
          {message.text}
        </div>
      )}
      
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="name">Full Name *</label>
          <input
            type="text"
            id="name"
            name="name"
            value={formData.name}
            onChange={handleInputChange}
            required
            placeholder="Enter full name"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="employee_id">Employee ID *</label>
          <input
            type="text"
            id="employee_id"
            name="employee_id"
            value={formData.employee_id}
            onChange={handleInputChange}
            required
            placeholder="Enter employee ID"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="department">Department</label>
          <input
            type="text"
            id="department"
            name="department"
            value={formData.department}
            onChange={handleInputChange}
            placeholder="Enter department"
          />
        </div>
        
        <button 
          type="submit" 
          disabled={isSubmitting || !formData.name || !formData.employee_id}
          className="btn btn-primary"
        >
          {isSubmitting ? '⏳ Registering...' : '✅ Register Person'}
        </button>
      </form>
      
      <div className="info-box">
        <h3>ℹ️ Note</h3>
        <p>
          After registering a person's details, you'll need to capture their face images 
          using the example_usage.py script in registration mode for the facial recognition 
          to work properly.
        </p>
        <code>
          python example_usage.py --mode register --name "John Doe" --employee-id "EMP001"
        </code>
      </div>
    </div>
  );
};

export default PersonRegistration;