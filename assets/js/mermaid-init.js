// Mermaid initialization for diagram rendering
// Supports dark/light theme based on Minimal Mistakes skin

// Initialize Mermaid (auto-detect theme)
mermaid.initialize({
  startOnLoad: false,
  theme: 'default'  // Use 'dark' for dark skins, 'default' for light
});

// Convert code blocks with language-mermaid to Mermaid diagrams
document.addEventListener('DOMContentLoaded', function() {
  // Find all code blocks with class 'language-mermaid'
  document.querySelectorAll('pre code.language-mermaid').forEach(function(el) {
    const pre = el.parentElement;
    const div = document.createElement('div');
    div.className = 'mermaid';
    div.textContent = el.textContent;

    // Replace pre with div
    pre.parentElement.replaceChild(div, pre);
  });

  // Render all Mermaid diagrams
  mermaid.run();
});
