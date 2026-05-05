// Configuration loader for node graph parameters
// Loads and parses config.yaml

let CONFIG = null;

async function loadConfig() {
  try {
    const response = await fetch('config.yaml');
    if (!response.ok) {
      console.error('Failed to load config.yaml:', response.statusText);
      return null;
    }
    const yamlText = await response.text();
    CONFIG = parseYAML(yamlText);
    console.log('Configuration loaded:', CONFIG);
    return CONFIG;
  } catch (error) {
    console.error('Error loading config:', error);
    return null;
  }
}

// Simple YAML parser (supports the subset we need)
function parseYAML(yamlText) {
  const lines = yamlText.split('\n');
  const result = {};
  const stack = [{ obj: result, indent: -1 }];
  
  for (let line of lines) {
    // Skip comments and empty lines
    const commentIndex = line.indexOf('#');
    if (commentIndex === 0) continue;
    if (commentIndex > 0) {
      line = line.substring(0, commentIndex);
    }
    if (!line.trim()) continue;
    
    // Calculate indentation
    const indent = line.search(/\S/);
    const trimmed = line.trim();
    
    // Pop stack to matching indent level
    while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
      stack.pop();
    }
    
    const current = stack[stack.length - 1].obj;
    
    // Parse key-value pair
    if (trimmed.includes(':')) {
      const colonIndex = trimmed.indexOf(':');
      const key = trimmed.substring(0, colonIndex).trim();
      const valueStr = trimmed.substring(colonIndex + 1).trim();
      
      if (!valueStr || valueStr === '{}') {
        // Empty object
        current[key] = valueStr === '{}' ? {} : {};
        stack.push({ obj: current[key], indent });
      } else if (valueStr.startsWith('[')) {
        // Array value
        current[key] = parseArray(valueStr);
      } else if (valueStr === 'true') {
        current[key] = true;
      } else if (valueStr === 'false') {
        current[key] = false;
      } else if (!isNaN(valueStr)) {
        current[key] = parseFloat(valueStr);
      } else {
        // String value (remove quotes if present)
        current[key] = valueStr.replace(/^["']|["']$/g, '');
      }
    }
  }
  
  return result;
}

function parseArray(str) {
  // Parse [1, 2, 3] format
  const content = str.slice(1, -1); // Remove [ ]
  if (!content.trim()) return [];
  return content.split(',').map(item => {
    const trimmed = item.trim();
    if (trimmed === 'true') return true;
    if (trimmed === 'false') return false;
    if (!isNaN(trimmed)) return parseFloat(trimmed);
    return trimmed.replace(/^["']|["']$/g, '');
  });
}

// Helper function to get parameter config for a node type and parameter name
function getParamConfig(nodeType, paramName) {
  if (!CONFIG || !CONFIG.nodes || !CONFIG.nodes[nodeType]) {
    return null;
  }
  return CONFIG.nodes[nodeType][paramName] || null;
}

// Helper function to get all parameters for a node type
function getNodeConfig(nodeType) {
  if (!CONFIG || !CONFIG.nodes) {
    return null;
  }
  return CONFIG.nodes[nodeType] || null;
}

// Helper function to get UI config
function getUIConfig(key) {
  if (!CONFIG || !CONFIG.ui) {
    return null;
  }
  return key ? CONFIG.ui[key] : CONFIG.ui;
}

// Helper function to get connection rules
function getConnectionRules(nodeCategory) {
  if (!CONFIG || !CONFIG.connections) {
    return null;
  }
  return CONFIG.connections[nodeCategory] || null;
}

// Helper function to get raycast config
function getRaycastConfig() {
  if (!CONFIG || !CONFIG.raycast) {
    return {
      'x-step': 10,
      'y-step': 10,
      'shadow-color': '#ff0000'
    };
  }
  return CONFIG.raycast;
}

// Helper function to get global settings
function getGlobalSetting(key, defaultValue) {
  if (!CONFIG) {
    return defaultValue;
  }
  return CONFIG[key] !== undefined ? CONFIG[key] : defaultValue;
}

// Helper function to set global settings
function setGlobalSetting(key, value) {
  if (!CONFIG) {
    CONFIG = {};
  }
  CONFIG[key] = value;
}
