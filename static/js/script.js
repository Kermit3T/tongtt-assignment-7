function validateHypothesisForm() {
    const parameter = document.getElementById('parameter').value;
    const testType = document.getElementById('test_type').value;
    
    if (!parameter || !testType) {
        alert('Please select both a parameter and test type');
        return false;
    }
    return true;
}

function validateConfidenceForm() {
    const parameter = document.getElementById('ci_parameter').value;
    const confidenceLevel = document.getElementById('confidence_level').value;
    
    if (!parameter || !confidenceLevel) {
        alert('Please select both a parameter and confidence level');
        return false;
    }
    return true;
}