// Theme Toggle Functionality
document.addEventListener('DOMContentLoaded', function() {
    const toggleSwitch = document.querySelector('#checkbox');
    const currentTheme = localStorage.getItem('theme') || 'dark-mode';
    
    // Set default to dark mode
    document.body.classList.add(currentTheme);
    
    // Update toggle position - INVERTED: checked for dark mode, unchecked for light mode
    if (currentTheme === 'dark-mode') {
        toggleSwitch.checked = true;
    } else {
        toggleSwitch.checked = false;
    }
    
    // Store the default theme if not already set
    if (!localStorage.getItem('theme')) {
        localStorage.setItem('theme', 'dark-mode');
    }
    
    // Function to switch theme - INVERTED logic
    function switchTheme(e) {
        if (e.target.checked) {
            document.body.classList.remove('light-mode');
            document.body.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
            document.body.classList.add('light-mode');
            localStorage.setItem('theme', 'light-mode');
        }
    }
    
    // Event listener for theme toggle
    toggleSwitch.addEventListener('change', switchTheme);
});
