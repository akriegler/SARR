import * as THREE from './lib/three/three.module.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

let pointSparsityStep = 10; // degrees (default from config)
let objectSparsityX = 2; // X-axis (alpha) object sparsity for SO(3) mode (default from config)
let objectSparsityY = -1; // Y-axis (beta) object sparsity for SO(3) mode (default from config)
let objectSparsityZ = 6; // Y-axis (beta) object sparsity for SO(3) mode (default from config)
let showPointCloud = true;
let showGrid = false;
let showObjects = true;
let showAxes = true;
let axisColor = 0xFFD700
let axisColorX= 0xFF0000
let axisColorY= 0x00FF00
let axisColorZ= 0x0000FF
let BOPSO3 = false; // if true, use full SO(3) range instead of BOP subset
let colorPalette = 'RGB';
let config = null;

loadConfig().then(cfg => {
  if (cfg && cfg.visualization) {
    config = cfg;
    console.log('Config loaded, applying visualization settings...');

    // Apply config defaults to global settings
    if (config.visualization.pointSparsity) {
      pointSparsityStep = config.visualization.pointSparsity.default || 10;
    }
    if (config.visualization.objectSparsityZ) {
      objectSparsityZ = config.visualization.objectSparsityZ.default || 4;
    }
    if (config.visualization.objectSparsityX) {
      objectSparsityX = config.visualization.objectSparsityX.default || 4;
    }
    if (config.visualization.objectSparsityY) {
      objectSparsityY = config.visualization.objectSparsityY.default || 4;
    }
    if (config.visualization.axisColorX) {
      axisColorX = parseInt(config.visualization.axisColorX.replace('#', '0x'));
    }
    if (config.visualization.axisColorY) {
      axisColorY = parseInt(config.visualization.axisColorY.replace('#', '0x'));
    }
    if (config.visualization.axisColorZ) {
      axisColorZ = parseInt(config.visualization.axisColorZ.replace('#', '0x'));
    }
    if (config.defaults) {
      showPointCloud = config.defaults.showPointCloud !== undefined ? config.defaults.showPointCloud : true;
      showGrid = config.defaults.showGrid !== undefined ? config.defaults.showGrid : false;
      showObjects = config.defaults.showObjects !== undefined ? config.defaults.showObjects : true;
      showAxes = config.defaults.showAxes !== undefined ? config.defaults.showAxes : true;
      BOPSO3 = config.defaults.BOPSO3 !== undefined ? config.defaults.BOPSO3 : false;
    }
    updateUIFromConfig();
  }
}).catch(err => console.warn('Config loading failed:', err));

function getSymmetryVector(symClass) {
  const symmetryVectors = {
    '1': [1, 1, 1],
    '2': [1, 1, 2],
    '3': [1, 1, 4],
    '4': [1, 1, 5],
    '5': [1, 1, 12],
    '6': [1, 1, 18],
    '7': [1, 1, 23],
    '8': [1, 1, 1000],
    '9': [1, 2, 1],
    '10': [2, 2, 2],
    '11': [2, 2, 1000]
  };
  return symmetryVectors[symClass] || [1, 1, 1];
}

function mod(a, n) {
  return ((a % n) + n) % n;
}

function clamp_rot_adv(alpha, beta, gamma, kappa) {
  if (kappa[0] === 2 && kappa[1] === 2 && kappa[2] === 2) {
    if (alpha > mod(alpha, Math.PI)) {
      alpha = mod(alpha, Math.PI);
      beta = mod((2 * Math.PI - beta), Math.PI);
      gamma = mod((2 * Math.PI - gamma), Math.PI);
    }
    else if (beta > mod(beta, Math.PI)) {
      alpha = mod(alpha, Math.PI);
      beta = mod(beta, Math.PI);
      gamma = mod((2 * Math.PI - gamma), Math.PI);
    }
    else {
        alpha = mod(alpha, Math.PI);
        beta = mod(beta, Math.PI);
        gamma = mod(gamma, Math.PI);
    }
  }
  else {
      alpha = (mod(alpha, (2 * Math.PI / kappa[0]))) * ((mod(kappa[0], 10 ** 3)) / kappa[0]);
      beta = (mod(beta, (2 * Math.PI / kappa[1]))) * ((mod(kappa[1], 10 ** 3)) / kappa[1]);
      gamma = (mod(gamma, (2 * Math.PI / kappa[2]))) * ((mod(kappa[2], 10 ** 3)) / kappa[2]);
    }
  if ((2 * Math.PI - 10**-5 <= alpha && 2* Math.PI + 10**-5 >= alpha) || (-(10**-5) <= alpha && 10**-5 >= alpha)){
    alpha = 0.0;
  }
  if ((2 * Math.PI - 10**-5 <= beta && 2* Math.PI + 10**-5 >= beta) || (-(10**-5) <= beta && 10**-5 >= beta)){
    beta = 0.0;
  }
  if ((2 * Math.PI - 10**-5 <= gamma && 2* Math.PI + 10**-5 >= gamma) || (-(10**-5) <= gamma && 10**-5 >= gamma)){
    gamma = 0.0;
  }

  return [alpha, beta, gamma];
}

function symAwareRotation(alpha, beta, gamma, symClass) {
  const kappa = getSymmetryVector(symClass);

  let angs = clamp_rot_adv(alpha, beta, gamma, kappa);
  alpha = angs[0];
  beta = angs[1];
  gamma = angs[2];
  const c_a = Math.cos(alpha);
  const c_b = Math.cos(beta);

  let s_a_, c_a_, s_b_, c_b_, s_g_, c_g_;

  // Class 1
  if (Math.max(...kappa) === 1) {
    s_a_ = Math.sin(alpha);
    c_a_ = Math.cos(alpha);
    s_b_ = Math.sin(beta);
    c_b_ = Math.cos(beta);
    s_g_ = Math.sin(gamma);
    c_g_ = Math.cos(gamma);
  }
  // Classes 2, 3, 4, 5, 6, 7, 8
  else if (kappa[2] > 1 && kappa[0] === 1 && kappa[1] === 1) {
    s_a_ = Math.sin(alpha);
    c_a_ = Math.cos(alpha);
    s_b_ = Math.sin(beta);
    c_b_ = Math.cos(beta);
    s_g_ = Math.sin(gamma * (mod(kappa[2], 1000)));
    c_g_ = Math.cos(gamma * (mod(kappa[2], 1000)));
  }
  // Class 9
  else if (kappa[1] > 1 && kappa[0] === 1 && kappa[2] === 1) {
    s_a_ = Math.sin(alpha);
    c_a_ = Math.cos(alpha);
    s_b_ = Math.sin(beta * (mod(kappa[1], 1000)));
    c_b_ = Math.cos(beta * (mod(kappa[1], 1000)));
    s_g_ = Math.sin(gamma) * c_b;
    c_g_ = Math.cos(gamma);
  }
  // Class 10, 11
  else {
    // Full multi-axis symmetry (no component is 1)
    s_a_ = Math.sin(alpha * (mod(kappa[0], 1000)));
    c_a_ = Math.cos(alpha * (mod(kappa[0], 1000)));
    s_b_ = Math.sin(beta * (mod(kappa[1], 1000))) * c_a;
    c_b_ = Math.cos(beta * (mod(kappa[1], 1000)));
    s_g_ = Math.sin(gamma * (mod(kappa[2], 1000))) * c_a * c_b;
    c_g_ = Math.cos(gamma * (mod(kappa[2], 1000)));
  }

  return {
    s_a: s_a_, c_a: c_a_,
    s_b: s_b_, c_b: c_b_,
    s_g: s_g_, c_g: c_g_
  };
}

// Generate rotation samples (matching Python code)
function generateRotationSamples(symClass, sparsityStep = 10) {
  const samples = [];

  if (BOPSO3) {
    // Full SO(3) cube: α ∈ [0, 2π), β ∈ [0, 2π), γ ∈ [0, 2π) (matching Python: np.arange(0, 360, step))
    for (let alphaDeg = 0; alphaDeg < 365; alphaDeg += sparsityStep) {
      for (let betaDeg = 0; betaDeg < 365; betaDeg += sparsityStep) {
        for (let gammaDeg = 0; gammaDeg < 365; gammaDeg += sparsityStep) {
          samples.push({
            alpha: alphaDeg * Math.PI / 180,
            beta: betaDeg * Math.PI / 180,
            gamma: gammaDeg * Math.PI / 180,
            set: 1
          });
        }
      }
    }
    return samples;
  }

  // TLESS sampling (original)
  // Two sets of alpha values (deg2rad conversion)
  // Alpha always starts at 5° and steps by 10° (matching Python: np.arange(5, 90, 10))
  const a1 = [];
  for (let deg = 5; deg < 90; deg += 10) {
    a1.push(deg * Math.PI / 180);
  }

  const a2 = [];
  if (symClass !== '9') {
    for (let deg = 275; deg < 360; deg += 10) {
      a2.push(deg * Math.PI / 180);
    }
  } else {
    a2.push(...a1);
  }

  // Beta and gamma ranges
  const b1 = [0];
  const b2 = [0];

  const g1 = [];
  const g2 = [];
  for (let deg = 0; deg < 360; deg += 5) {
    const rad = deg * Math.PI / 180;
    g1.push(rad);
    g2.push(rad);
  }

  // Generate cartesian product for set 1
  for (const alpha of a1) {
    for (const beta of b1) {
      for (const gamma of g1) {
        samples.push({ alpha, beta, gamma, set: 1 });
      }
    }
  }

  // Generate cartesian product for set 2
  for (const alpha of a2) {
    for (const beta of b2) {
      for (const gamma of g2) {
        samples.push({ alpha, beta, gamma, set: 2 });
      }
    }
  }

  return samples;
}

// Viridis colorscale (approximation)
function viridiscolor(t) {
  // Clamp t to [-1, 1] and map to [0, 1]
  t = (t + 1) / 2;
  t = Math.max(0, Math.min(1, t));

  // Better viridis approximation
  const colors = [
    [0.267004, 0.004874, 0.329415],
    [0.282623, 0.140926, 0.457517],
    [0.253935, 0.265254, 0.529983],
    [0.206756, 0.371758, 0.553117],
    [0.163625, 0.471133, 0.558148],
    [0.127568, 0.566949, 0.550556],
    [0.134692, 0.658636, 0.517649],
    [0.266941, 0.748751, 0.440573],
    [0.477504, 0.821444, 0.318195],
    [0.741388, 0.873449, 0.149561],
    [0.993248, 0.906157, 0.143936]
  ];

  const idx = Math.floor(t * (colors.length - 1));
  const nextIdx = Math.min(idx + 1, colors.length - 1);
  const localT = (t * (colors.length - 1)) - idx;

  const c1 = colors[idx];
  const c2 = colors[nextIdx];

  const r = c1[0] + (c2[0] - c1[0]) * localT;
  const g = c1[1] + (c2[1] - c1[1]) * localT;
  const b = c1[2] + (c2[2] - c1[2]) * localT;

  return new THREE.Color(r, g, b);
}

const COLOR_PALETTES = {
  RGB: {
    p_x: [255, 0, 0],         // Pure red
    n_x: [100, 0, 0],         // Deep red
    p_y: [0, 255, 0],         // Pure green
    n_y: [0, 100, 0],          // Deep green
    p_z: [0, 0, 255],      // Vivid blue
    n_z: [0, 0, 100]         // Deep blue
   },
  GRAY: {
    p_x: [130, 130, 130],         // Pure red
    n_x: [130, 130, 130],         // Deep red
    p_y: [130, 130, 130],         // Pure green
    n_y: [130, 130, 130],          // Deep green
    p_z: [130, 130, 130],      // Vivid blue
    n_z: [130, 130, 130],         // Deep blue
   },
  VIRIDIS: {
    p_z: [255, 255, 0],       // Pure yellow (max saturation)
    n_z: [128, 0, 255],       // Pure purple (max saturation)
    p_x: [0, 255, 100],       // Bright green (high saturation)
    p_y: [0, 200, 200],       // Bright teal (high saturation)
    n_x: [0, 100, 255],       // Bright blue (high saturation)
    n_y: [200, 255, 0]        // Lime yellow (high saturation)
  },
  PLASMA: {
    p_z: [255, 255, 0],       // Pure yellow (max saturation)
    n_z: [100, 0, 255],       // Deep purple (max saturation)
    p_x: [255, 0, 150],       // Bright magenta (high saturation)
    p_y: [0, 255, 100],       // Bright green (high saturation)
    n_x: [150, 0, 255],       // Blue-purple (high saturation)
    n_y: [255, 200, 0]        // Orange-yellow (high saturation)
  },
  INFERNO: {
    p_z: [255, 255, 100],     // Bright yellow (high saturation)
    n_z: [50, 0, 50],         // Deep purple-black (high contrast)
    p_x: [255, 0, 50],        // Bright red (max saturation)
    p_y: [150, 0, 200],       // Purple (high saturation)
    n_x: [255, 100, 0],       // Bright orange (max saturation)
    n_y: [255, 255, 50]       // Light yellow (high saturation)
  },
  MAGMA: {
    p_z: [255, 255, 150],     // Light yellow (high saturation)
    n_z: [50, 0, 50],         // Deep purple-black (high contrast)
    p_x: [255, 50, 100],      // Red-orange (high saturation)
    p_y: [200, 0, 200],       // Magenta (max saturation)
    n_x: [255, 150, 0],       // Orange (max saturation)
    n_y: [100, 200, 255]      // Light blue (high saturation)
  },
  CIVIDIS: {
    p_z: [255, 255, 0],       // Pure yellow (max saturation)
    n_z: [0, 50, 150],        // Dark blue (high saturation)
    p_x: [0, 200, 150],       // Teal (high saturation)
    p_y: [200, 200, 0],       // Olive-yellow (high saturation)
    n_x: [0, 255, 255],       // Cyan (max saturation)
    n_y: [255, 220, 0]        // Yellow (high saturation)
  },
  TWILIGHT: {
    p_z: [255, 100, 255],     // Bright magenta (max saturation)
    n_z: [200, 150, 255],     // Light purple (high saturation)
    p_x: [150, 0, 150],       // Dark magenta (max saturation)
    p_y: [100, 200, 0],       // Lime green (high saturation)
    n_x: [255, 50, 200],      // Pink (high saturation)
    n_y: [50, 150, 255]       // Sky blue (high saturation)
  },
  RAINBOW: {
    p_z: [148, 0, 255],       // Purple (max saturation)
    n_z: [255, 0, 0],         // Red (max saturation)
    p_x: [0, 255, 0],         // Green (max saturation)
    p_y: [0, 255, 255],       // Cyan (max saturation)
    n_x: [255, 255, 0],       // Yellow (max saturation)
    n_y: [255, 128, 0]        // Orange (max saturation)
  },
  JET: {
    p_z: [255, 0, 0],         // Red (max saturation)
    n_z: [0, 0, 200],         // Dark blue (high saturation)
    p_x: [0, 255, 255],       // Cyan (max saturation)
    p_y: [255, 255, 0],       // Yellow (max saturation)
    n_x: [0, 150, 255],       // Light blue (high saturation)
    n_y: [255, 128, 0]        // Orange (max saturation)
  },
  SEISMIC: {
    p_z: [0, 100, 255],       // Blue (high saturation)
    n_z: [255, 0, 0],         // Red (max saturation)
    p_x: [255, 255, 255],     // White (max brightness)
    p_y: [100, 150, 255],     // Light blue (high saturation)
    n_x: [255, 100, 100],     // Light red (high saturation)
    n_y: [200, 200, 200]      // Light gray
  },
  COOLWARM: {
    p_z: [255, 0, 50],        // Bright red (max saturation)
    n_z: [0, 100, 255],       // Bright blue (max saturation)
    p_x: [240, 240, 240],     // Light gray (high brightness)
    p_y: [255, 120, 0],       // Orange (max saturation)
    n_x: [0, 200, 255],       // Bright cyan (high saturation)
    n_y: [255, 150, 100]      // Light orange (high saturation)
  }
};

// Get colors for the current palette
function getPaletteColors(paletteName = 'RGB') {
  const palette = COLOR_PALETTES[paletteName] || COLOR_PALETTES.default;
  const colors = {};
  for (const [key, rgb] of Object.entries(palette)) {
    colors[key] = new THREE.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255);
  }
  return colors;
}

function getConeGradientColors(paletteName = 'RGB', numColors = 16) {
  if (paletteName === 'RGB') {
    return [
      [255, 115, 0],
      [250, 135, 0],
      [235, 155, 0],
      [215, 175, 0],
      [195, 195, 0],
      [175, 215, 0],
      [155, 235, 0],
      [135, 250, 0],     // Pure red
      [125, 255, 0],
      [115, 250, 0],    // Red-orange
      [95, 235, 0],   // Orange
      [75, 215, 0],   // Yellow-orange
      [55, 195, 0],
      [35, 175, 0],   // Yellow-green
      [15, 155, 0],   // Yellow-green
      [5, 135, 0],
      [0, 115, 0],   // Yellow-green
      [5, 95, 0],
      [15, 75, 0],     // Pure green
      [35, 55, 0],   // Green-cyan
      [55, 35, 0],   // Cyan
      [75, 15, 0],   // Cyan-blue
      [95, 5, 0],     // Pure blue
      [115, 0, 0],   // Blue-purple
      [125, 5, 0],
      [135, 15, 0],
      [155, 35, 0],
      [175, 55, 0],   // Magenta
      [195, 75, 0],   // Magenta-red
      [215, 95, 0],
      [235, 115, 0],     // Dark red
      [255, 135, 0],    // Bright orange
    ];
  }
  else if (paletteName === 'GRAY') {
        return [
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
          [130, 130, 130],
      ];
  }
  const palette = COLOR_PALETTES[paletteName] || COLOR_PALETTES.default;
  const gradientColors = [];

  // For other palettes, interpolate between all 6 face colors
  const paletteColors = [
    palette.p_z, palette.p_x, palette.p_y,
    palette.n_z, palette.n_x, palette.n_y
  ];

  for (let i = 0; i < numColors; i++) {
    const t = i / (numColors - 1);
    const segmentCount = paletteColors.length - 1;
    const segment = Math.min(Math.floor(t * segmentCount), segmentCount - 1);
    const localT = (t * segmentCount) - segment;

    const c1 = paletteColors[segment];
    const c2 = paletteColors[segment + 1];

    gradientColors.push([
      Math.round(c1[0] + (c2[0] - c1[0]) * localT),
      Math.round(c1[1] + (c2[1] - c1[1]) * localT),
      Math.round(c1[2] + (c2[2] - c1[2]) * localT)
    ]);
  }

  return gradientColors;
}

function createObjectPrimitive(symClass) {
  const group = new THREE.Group();

  // Get colors from current palette
  const colors = getPaletteColors(colorPalette);

  let vertices, faces, facecolors;

  if (symClass === '1') {
    vertices = [
      // Main box vertices (0-7)
      [-0.25, -0.5, -0.75], [0.25, -0.5, -0.75], [0.25, 0.5, -0.75], [-0.25, 0.5, -0.75],
      [-0.25, -0.5, 0.75], [0.25, -0.5, 0.75], [0.25, 0.5, 0.75], [-0.25, 0.5, 0.75],
      // Middle extension vertices (8-15)
      [-0.125, -0.25, 0], [0.125, -0.25, 0], [0.125, 0.25, 0], [-0.125, 0.25, 0],
      [-0.125, -0.25, 1.5], [0.125, -0.25, 1.5], [0.125, 0.25, 1.5], [-0.125, 0.25, 1.5],
      // Top extension vertices (16-23)
      [-0.125, 0, -0.375], [0.125, 0, -0.375], [0.125, 1, -0.375], [-0.125, 1, -0.375],
      [-0.125, 0, 0.375], [0.125, 0, 0.375], [0.125, 1, 0.375], [-0.125, 1, 0.375]
    ];

    // Define faces as triplets of vertex indices
    faces = [
      // Main box faces (all 6 sides)
      [0,1,2], [0,2,3],   // -Z face
      [4,6,5], [4,7,6],   // +Z face
      [0,4,5], [0,5,1],   // -Y face
      [2,6,7], [2,7,3],   // +Y face
      [0,3,7], [0,7,4],   // -X face
      [1,5,6], [1,6,2],   // +X face
      // Middle extension (all 6 sides)
      [8,9,10], [8,10,11],     // Bottom
      [12,14,13], [12,15,14],  // Top
      [8,12,13], [8,13,9],     // -Y face
      [10,14,15], [10,15,11],  // +Y face
      [8,11,15], [8,15,12],    // -X face
      [9,13,14], [9,14,10],    // +X face
      // Top extension (all 6 sides)
      [16,17,18], [16,18,19],  // Bottom
      [20,22,21], [20,23,22],  // Top
      [16,20,21], [16,21,17],  // -Y face
      [18,22,23], [18,23,19],  // +Y face
      [16,19,23], [16,23,20],  // -X face
      [17,21,22], [17,22,18]   // +X face
    ];

    facecolors = [
      // Main box
      colors.n_z, colors.n_z,  // -Z (2 triangles)
      colors.p_z, colors.p_z,  // +Z
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x,  // +X
      // Middle extension
      colors.n_z, colors.n_z,  // Bottom
      colors.p_z, colors.p_z,  // Top
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x,  // +X
      // Top extension
      colors.n_z, colors.n_z,  // Bottom
      colors.p_z, colors.p_z,  // Top
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x   // +X
    ];

  }
  else if (symClass === '2') {
    // L-shaped object with extensions
    vertices = [
      // Main box vertices (0-7)
      [-0.25, -0.5, -0.75], [0.25, -0.5, -0.75], [0.25, 0.5, -0.75], [-0.25, 0.5, -0.75],
      [-0.25, -0.5, 0.75], [0.25, -0.5, 0.75], [0.25, 0.5, 0.75], [-0.25, 0.5, 0.75],
      // Extension vertices (8-15)
      [-0.125, -0.25, 0], [0.125, -0.25, 0], [0.125, 0.25, 0], [-0.125, 0.25, 0],
      [-0.125, -0.25, 1.5], [0.125, -0.25, 1.5], [0.125, 0.25, 1.5], [-0.125, 0.25, 1.5]
    ];

    faces = [
      // Main box (all 6 sides)
      [0,1,2], [0,2,3],   // -Z face
      [4,6,5], [4,7,6],   // +Z face
      [0,4,5], [0,5,1],   // -Y face
      [2,6,7], [2,7,3],   // +Y face
      [0,3,7], [0,7,4],   // -X face
      [1,5,6], [1,6,2],   // +X face
      // Extension (all 6 sides)
      [8,9,10], [8,10,11],     // Bottom
      [12,14,13], [12,15,14],  // Top
      [8,12,13], [8,13,9],     // -Y face
      [10,14,15], [10,15,11],  // +Y face
      [8,11,15], [8,15,12],    // -X face
      [9,13,14], [9,14,10]     // +X face
    ];

    facecolors = [
      // Main box
      colors.n_z, colors.n_z,  // -Z
      colors.p_z, colors.p_z,  // +Z
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x,  // +X
      // Extension
      colors.n_z, colors.n_z,  // Bottom
      colors.p_z, colors.p_z,  // Top
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x   // +X
    ];
  }
  else if (symClass === '3' || symClass === '4' || symClass === '5' || symClass === '6' || symClass === '7' || symClass === '8') {
    let numPoints;

    // Cone shape for continuous rotation symmetry
    if (symClass === '3') {
      numPoints = 4;
    }
    else if (symClass === '4') {
      numPoints = 5;
    }
    else if (symClass === '5') {
      numPoints = 12;
    }
    else if (symClass === '6') {
      numPoints = 18;
    }
    else if (symClass === '7') {
      numPoints = 23;
    }
    else if (symClass === '8') {
      numPoints = 100;
    }
    const height = 2.0;
    const radius = 0.4;

    vertices = [[0, 0, height/2]]; // Apex

    // Base vertices
    const angleStep = 2 * Math.PI / numPoints;
    for (let i = 0; i < numPoints; i++) {
      const angle = i * angleStep;
      vertices.push([
        radius * Math.cos(angle),
        radius * Math.sin(angle),
        -height/2
      ]);
    }
    vertices.push([0, 0, -height/2]); // Base center

    faces = [];
    facecolors = [];

    // Get gradient colors for cone based on current palette
    const gradientcolors = getConeGradientColors(colorPalette, numPoints);

    // Base faces
    for (let i = 0; i < numPoints; i++) {
      const v0 = i + 1;
      const v1 = (i + 1) % numPoints + 1;
      const v2 = numPoints + 1;
      faces.push([v0, v1, v2]);
      if (colorPalette === 'GRAY') {
        facecolors.push(new THREE.Color(0.5, 0.5, 0.5))
        }else {
        facecolors.push(new THREE.Color(0.0, 0.0, 0.33))
      }
    }

    // Side faces
    for (let i = 0; i < numPoints; i++) {
      const v0 = i + 1;
      const v1 = (i + 1) % numPoints + 1;
      const v2 = 0;
      faces.push([v0, v2, v1]);
      const rgb = gradientcolors[Math.floor(i * (gradientcolors.length >= numPoints ? Math.floor((gradientcolors.length) / numPoints) : 0.32))];
      facecolors.push(new THREE.Color(rgb[0]/255, rgb[1]/255, rgb[2]/255));
    }
  }
  else if (symClass === '9') {
    vertices = [
      // Main box vertices (0-7)
      [-0.25, -0.5, -0.75], [0.25, -0.5, -0.75], [0.25, 0.5, -0.75], [-0.25, 0.5, -0.75],
      [-0.25, -0.5, 0.75], [0.25, -0.5, 0.75], [0.25, 0.5, 0.75], [-0.25, 0.5, 0.75],
      // Extension vertices (8-15)
      [-0.125, 0, -0.375], [0.125, 0, -0.375], [0.125, 1, -0.375], [-0.125, 1, -0.375],
      [-0.125, 0, 0.375], [0.125, 0, 0.375], [0.125, 1, 0.375], [-0.125, 1, 0.375]
    ];

    vertices = vertices.map(v => [v[0], v[1], v[2]]);

    faces = [
      // Main box (all 6 sides)
      [0,1,2], [0,2,3],   // -Z face
      [4,6,5], [4,7,6],   // +Z face
      [0,4,5], [0,5,1],   // -Y face
      [2,6,7], [2,7,3],   // +Y face
      [0,3,7], [0,7,4],   // -X face
      [1,5,6], [1,6,2],   // +X face
      // Extension (all 6 sides)
      [8,9,10], [8,10,11],     // Bottom
      [12,14,13], [12,15,14],  // Top
      [8,12,13], [8,13,9],     // -Y face
      [10,14,15], [10,15,11],  // +Y face
      [8,11,15], [8,15,12],    // -X face
      [9,13,14], [9,14,10]     // +X face
    ];

    facecolors = [
      // Main box
      colors.n_z, colors.n_z,  // -Z
      colors.p_z, colors.p_z,  // +Z
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x,  // +X
      // Extension
      colors.n_z, colors.n_z,  // Bottom
      colors.p_z, colors.p_z,  // Top
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x   // +X
    ];

  }
  else if (symClass === '10') {
    // Default simple box
    vertices = [
      [-0.25, -0.5, -0.75], [0.25, -0.5, -0.75], [0.25, 0.5, -0.75], [-0.25, 0.5, -0.75],
      [-0.25, -0.5, 0.75], [0.25, -0.5, 0.75], [0.25, 0.5, 0.75], [-0.25, 0.5, 0.75]
    ];

    faces = [
      [0,1,2], [0,2,3], [4,5,6], [4,6,7],
      [0,1,5], [0,5,4], [2,3,7], [2,7,6],
      [0,3,7], [0,7,4], [1,2,6], [1,6,5]
    ];

    facecolors = [
      // Main box
      colors.n_z, colors.n_z,  // -Z (2 triangles)
      colors.p_z, colors.p_z,  // +Z
      colors.n_y, colors.n_y,  // -Y
      colors.p_y, colors.p_y,  // +Y
      colors.n_x, colors.n_x,  // -X
      colors.p_x, colors.p_x,  // +X
    ];
  } else if (symClass === '11') {

    const numPoints = 100;
    const height = 2.0;
    const radius = 0.4;

    vertices = [];

    const topZ = height / 2;
    const bottomZ = -height / 2;
    const angleStep = 2 * Math.PI / numPoints;

    // --- Top center
    const topCenter = vertices.length;
    vertices.push([0, 0, topZ]);

    // --- Top ring
    const topStart = vertices.length;
    for (let i = 0; i < numPoints; i++) {
      const angle = i * angleStep;
      vertices.push([
        radius * Math.cos(angle),
        radius * Math.sin(angle),
        topZ
      ]);
    }

    // --- Bottom center
    const bottomCenter = vertices.length;
    vertices.push([0, 0, bottomZ]);

    // --- Bottom ring
    const bottomStart = vertices.length;
    for (let i = 0; i < numPoints; i++) {
      const angle = i * angleStep;
      vertices.push([
        radius * Math.cos(angle),
        radius * Math.sin(angle),
        bottomZ
      ]);
    }

    faces = [];
    facecolors = [];

    const gradientcolors = getConeGradientColors(colorPalette, numPoints);

    // --- Top cap
    for (let i = 0; i < numPoints; i++) {
      const v0 = topStart + i;
      const v1 = topStart + (i + 1) % numPoints;
      const v2 = topCenter;

      faces.push([v0, v2, v1]);
      if (colorPalette === 'GRAY') {
        facecolors.push(new THREE.Color(0.5, 0.5,0.5))
        }else {
        facecolors.push(new THREE.Color(0.0, 0.0, 1.0))
      }
      }


    // --- Bottom cap
    for (let i = 0; i < numPoints; i++) {
      const v0 = bottomStart + i;
      const v1 = bottomStart + (i + 1) % numPoints;
      const v2 = bottomCenter;

      faces.push([v0, v1, v2]);

      if (colorPalette === 'GRAY') {
        facecolors.push(new THREE.Color(0.5, 0.5, 0.5))
        }else {
        facecolors.push(new THREE.Color(0.0, 0.0, 0.33))
      }
    }

    // --- Side walls (2 triangles per segment)
    for (let i = 0; i < numPoints; i++) {
      const t0 = topStart + i;
      const t1 = topStart + (i + 1) % numPoints;

      const b0 = bottomStart + i;
      const b1 = bottomStart + (i + 1) % numPoints;

      // triangle 1
      faces.push([t0, b0, t1]);
      facecolors.push(new THREE.Color(...gradientcolors[Math.floor(i * (gradientcolors.length >= numPoints ? Math.floor((gradientcolors.length) / numPoints) : 0.32))].map(v => v/255)));

      // triangle 2
      faces.push([t1, b0, b1]);
      facecolors.push(new THREE.Color(...gradientcolors[Math.floor(i * (gradientcolors.length >= numPoints ? Math.floor((gradientcolors.length) / numPoints) : 0.32))].map(v => v/255)));
    }
  }

  // Create geometry
  const geometry = new THREE.BufferGeometry();
  const positions = [];
  const facecolorArray = [];

  // Build triangles
  for (let i = 0; i < faces.length; i++) {
    const face = faces[i];
    const color = facecolors[i] || new THREE.Color(0.5, 0.5, 0.5);

    for (let j = 0; j < 3; j++) {
      const v = vertices[face[j]];
      positions.push(v[0], v[1], v[2]);
      facecolorArray.push(color.r, color.g, color.b);
    }
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(facecolorArray, 3));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    flatShading: true,
    metalness: 0.2,
    roughness: 0.7,
    side: THREE.DoubleSide
  });

  const mesh = new THREE.Mesh(geometry, material);
  group.add(mesh);

  // Scale down the entire primitive
  group.scale.setScalar(0.2);

  return group;
}

class InsetViewer {
  constructor() {
    this.container = document.getElementById('insetViewer');
    this.canvasContainer = document.getElementById('insetCanvas');

    const size = 200;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(size, size);
    this.renderer.setClearColor(0x000000, 0);
    this.canvasContainer.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    this.camera.position.set(2.8, 2.2, 2.8);
    this.camera.lookAt(0, 0, 0);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;
    this.controls.enableZoom = false;
    this.controls.enablePan = false;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.0;
    this.controls.target.set(0, 0, 0);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(3, 4, 5);
    this.scene.add(dirLight);

    this.objectGroup = new THREE.Group();
    this.axesGroup = new THREE.Group();
    this.scene.add(this.objectGroup);
    this.scene.add(this.axesGroup);

    this.labels = [];

    this.currentSymClass = null;
    this.visible = false;

    this._buildRotationAxes();
  }

  _buildRotationAxes() {
    // Clear existing
    while (this.axesGroup.children.length > 0) {
      const child = this.axesGroup.children[0];
      this.axesGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    }

    const axisLength = 1.6;
    const axisRadius = 0.018;

    // Axis colors: X=red (alpha), Y=green (beta), Z=blue (gamma)
    const axisConfigs = [
      { color: axisColorY, dir: new THREE.Vector3(1, 0, 0), label: 'α', arcNormal: new THREE.Vector3(1, 0, 0) },
      { color: axisColorZ, dir: new THREE.Vector3(0, 1, 0), label: 'β', arcNormal: new THREE.Vector3(0, 1, 0) },
      { color: axisColorX, dir: new THREE.Vector3(0, 0, 1), label: 'γ', arcNormal: new THREE.Vector3(0, 0, 1) },
    ];

    for (const cfg of axisConfigs) {
      // Straight axis line
      const lineMat = new THREE.MeshBasicMaterial({ color: cfg.color, transparent: true, opacity: 0.7 });
      const lineGeo = new THREE.CylinderGeometry(axisRadius, axisRadius, axisLength * 2, 8);
      const lineMesh = new THREE.Mesh(lineGeo, lineMat);
      // Orient cylinder along the axis direction
      lineMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), cfg.dir);
      this.axesGroup.add(lineMesh);

      // Arrowhead
      const arrowGeo = new THREE.ConeGeometry(0.05, 0.14, 8);
      const arrowMesh = new THREE.Mesh(arrowGeo, lineMat.clone());
      arrowMesh.position.copy(cfg.dir.clone().multiplyScalar(axisLength));
      arrowMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), cfg.dir);
      this.axesGroup.add(arrowMesh);

      // Rotation arc (partial torus showing rotation direction)
      const arcRadius = 0.75;
      const tubeRadius = 0.015;
      const arcAngle = Math.PI * 0.75; // 135° arc
      const arcGeo = new THREE.TorusGeometry(arcRadius, tubeRadius, 8, 32, arcAngle);
      const arcMat = new THREE.MeshBasicMaterial({ color: cfg.color, transparent: true, opacity: 0.5 });
      const arcMesh = new THREE.Mesh(arcGeo, arcMat);

      // Orient arc so it wraps around the axis
      if (cfg.label === 'α') {
        // X-axis rotation: arc in YZ plane
        arcMesh.rotation.set(0, Math.PI / 2, 0);
      } else if (cfg.label === 'β') {
        // Y-axis rotation: arc in XZ plane
        arcMesh.rotation.set(-Math.PI / 2, 0, 0);
      } else {
        arcMesh.rotation.set(0, 0, 0);
      }
      this.axesGroup.add(arcMesh);

      // Arc arrowhead (small cone at arc end to indicate rotation direction)
      const arcArrowGeo = new THREE.ConeGeometry(0.04, 0.1, 6);
      const arcArrowMat = new THREE.MeshBasicMaterial({ color: cfg.color, transparent: true, opacity: 0.7 });
      const arcArrowMesh = new THREE.Mesh(arcArrowGeo, arcArrowMat);

      // Position at end of arc
      const endAngle = arcAngle;
      const ax = arcRadius * Math.cos(endAngle);
      const ay = arcRadius * Math.sin(endAngle);

      if (cfg.label === 'α') {
        arcArrowMesh.position.set(0, ay, -ax);
        arcArrowMesh.rotation.set(0, 0, endAngle + Math.PI / 2);
      } else if (cfg.label === 'β') {
        arcArrowMesh.position.set(ax, 0, -ay);
        arcArrowMesh.rotation.set(-(endAngle + Math.PI / 2), 0, 0);
      } else {
        arcArrowMesh.position.set(ax, ay, 0);
        arcArrowMesh.rotation.set(0, 0, endAngle + Math.PI / 2);
      }
      this.axesGroup.add(arcArrowMesh);
    }
  }

  _clearLabels() {
    for (const lbl of this.labels) {
      if (lbl.element && lbl.element.parentNode) {
        lbl.element.parentNode.removeChild(lbl.element);
      }
    }
    this.labels = [];
  }

  _createLabels() {
    this._clearLabels();
    const labelConfigs = [
      { text: 'β', color: '#00ff00', position: new THREE.Vector3(1.85, 0, 0) },
      { text: 'γ', color: '#0000ff', position: new THREE.Vector3(0, 1.85, 0) },
      { text: 'α', color: '#ff0000', position: new THREE.Vector3(0, 0, 1.85) },
    ];

    for (const cfg of labelConfigs) {
      const el = document.createElement('div');
      el.className = 'inset-label';
      el.textContent = cfg.text;
      el.style.color = cfg.color;
      this.canvasContainer.appendChild(el);
      this.labels.push({ element: el, position: cfg.position });
    }
  }

  show(parentContainer, symClass) {
    this.container.classList.add('active');
    this.visible = true;

    // Update object if symmetry class changed
    if (this.currentSymClass !== symClass) {
      this.currentSymClass = symClass;
      // Clear old object
      while (this.objectGroup.children.length > 0) {
        const child = this.objectGroup.children[0];
        this.objectGroup.remove(child);
        child.traverse(c => {
          if (c.geometry) c.geometry.dispose();
          if (c.material) c.material.dispose();
        });
      }
      // Add new object
      const obj = createObjectPrimitive(symClass);
      obj.scale.setScalar(1.0); // override the 0.2 scale from createObjectPrimitive
      this.objectGroup.add(obj);
    }

    // Create labels
    this._createLabels();
  }

  hide() {
    this.container.classList.remove('active');
    this.visible = false;
    this._clearLabels();
  }

  render() {
    if (!this.visible) return;

    this.controls.update();
    this.renderer.render(this.scene, this.camera);

    // Update label positions (project 3D to 2D)
    const w = this.canvasContainer.clientWidth;
    const h = this.canvasContainer.clientHeight;
    if (w === 0 || h === 0) return;

    for (const lbl of this.labels) {
      const projected = lbl.position.clone().project(this.camera);
      const x = (projected.x * 0.5 + 0.5) * w;
      const y = (-projected.y * 0.5 + 0.5) * h;

      // Hide if behind camera
      if (projected.z > 1) {
        lbl.element.style.display = 'none';
      } else {
        lbl.element.style.display = 'block';
        lbl.element.style.left = x + 'px';
        lbl.element.style.top = y + 'px';
      }
    }
  }
}

// Global inset viewer instance
const insetViewer = new InsetViewer();

class SubplotViewer {
  constructor(containerId, paramName, index) {
    this.containerId = containerId;
    this.paramName = paramName;
    this.index = index;
    this.container = document.getElementById(containerId);

    if (!this.container) {
      console.error(`Container ${containerId} not found`);
      return;
    }

    // Setup renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.domElement.classList.add('webgl');
    this.container.appendChild(this.renderer.domElement);

    // Setup scene
    this.scene = new THREE.Scene();

    // Setup camera
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 1000);
    // Position camera to view the parameter space
    const centerX = Math.PI;
    const centerY = 0;
    const centerZ = Math.PI;
    const distance = 8;
    this.camera.position.set(centerX + distance, centerY + distance, centerZ + distance);
    this.camera.up.set(0, 0, 1);

    // Setup controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = false;
    // Set target to center of parameter space
    this.controls.target.set(centerX, centerY, centerZ);
    this.controls.update();

    // Parameter space bounds (alpha: 0-2π, beta: 0, gamma: 0-2π)
    this.bounds = {
      min: new THREE.Vector3(0, 0, 0),
      max: new THREE.Vector3(2 * Math.PI, 0.2, 2 * Math.PI)
    };

    // Add grid spanning full parameter range
    const gridSizeX = 2 * Math.PI;
    const gridSizeZ = 2 * Math.PI;
    const gridDivisions = 20;
    this.grid = new THREE.GridHelper(Math.max(gridSizeX, gridSizeZ), gridDivisions, 0xaaaaaa, 0xcccccc);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.4;
    // Center grid at parameter space center
    this.grid.position.set(Math.PI, 0, Math.PI);
    this.scene.add(this.grid);

    // Add axes at origin
    this.axes = new THREE.AxesHelper(1);
    this.scene.add(this.axes);

    // Add wireframe bounding box
    //this.boundsBox = this.createBoundsBox();
    //this.scene.add(this.boundsBox);

    // Add parameter axes
    this.parameterAxes = this.createParameterAxes();
    this.parameterAxes.visible = showAxes;
    this.scene.add(this.parameterAxes);

    // Add lights
    const ambient = new THREE.AmbientLight(0xffffff, 0.6);
    const dir = new THREE.DirectionalLight(0xffffff, 0.4);
    dir.position.set(5, 5, 5);
    this.scene.add(ambient, dir);

    // Point cloud for rotation samples
    this.pointCloud = null;

    // Object primitives (geometric shapes)
    this.objectPrimitives = [];

    // Data bounds (will be computed from actual samples)
    this.dataBounds = {
      min: new THREE.Vector3(Infinity, Infinity, Infinity),
      max: new THREE.Vector3(-Infinity, -Infinity, -Infinity)
    };

    // Orbit animation state
    this.autoOrbit = false;
    this.orbitAngle = 0;
    this.orbitSpeed = 0.3; // degrees per frame
    this.orbitRadius = 8;
    this.orbitCenter = new THREE.Vector3(centerX, centerY, centerZ);
  }

  createParameterAxes() {
    // Create axes along the edges of the parameter space bounds
    const group = new THREE.Group();
    const min = this.bounds.min;
    const max = this.bounds.max;

    // Axis material - uses configured color for better visibility
    const axisMaterialX = new THREE.MeshBasicMaterial({
      color: axisColorX
    });
    const axisMaterialY = new THREE.MeshBasicMaterial({
      color: axisColorY
    });
    const axisMaterialZ = new THREE.MeshBasicMaterial({
      color: axisColorZ
    });
    const axisRadius = 0.03; // Thick radius for visibility

    // Helper function to create a thick axis line using cylinder
    const createAxisLine = (start, end, ax) => {
      const direction = new THREE.Vector3().subVectors(end, start);
      const length = direction.length();
      const cylinder = new THREE.CylinderGeometry(axisRadius, axisRadius, length, 8);
      const material = ax === 'X' ? axisMaterialX : ax === 'Y' ? axisMaterialY : axisMaterialZ;

      const mesh = new THREE.Mesh(cylinder, material);

      // Position and orient the cylinder
      mesh.position.copy(start).add(direction.multiplyScalar(0.5));
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        direction.normalize()
      );

      return mesh;
    };

    // X-axis (alpha) - along bottom-front edge
    group.add(createAxisLine(
      new THREE.Vector3(min.x, min.y, min.z),
      new THREE.Vector3(max.x, min.y, min.z),
        'X'
    ));

    // Y-axis (beta) - along left-front edge
    group.add(createAxisLine(
      new THREE.Vector3(min.x, min.y, min.z),
      new THREE.Vector3(min.x, max.y, min.z),
        'Y'
    ));

    // Z-axis (gamma) - along left-bottom edge
    group.add(createAxisLine(
      new THREE.Vector3(min.x, min.y, min.z),
      new THREE.Vector3(min.x, min.y, max.z),
        'Z'
    ));

    // Store label data for DOM-based LaTeX rendering
    const labelOffset = 0.3;
    this.axisLabels = [
      { text: '$\\alpha$', position: new THREE.Vector3(max.x + labelOffset, min.y, min.z) },
      { text: '$\\beta$', position: new THREE.Vector3(min.x, max.y + labelOffset, min.z) },
      { text: '$\\gamma$', position: new THREE.Vector3(min.x, min.y, max.z + labelOffset) }
    ];

    return group;
  }

  updateVisualization(symClass) {
    // Remove existing point cloud
    if (this.pointCloud) {
      this.scene.remove(this.pointCloud);
      this.pointCloud.geometry.dispose();
      this.pointCloud.material.dispose();
    }

    // Remove existing object primitives
    for (const obj of this.objectPrimitives) {
      this.scene.remove(obj);
      obj.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
    }
    this.objectPrimitives = [];

    // Reset data bounds
    this.dataBounds.min.set(Infinity, Infinity, Infinity);
    this.dataBounds.max.set(-Infinity, -Infinity, -Infinity);

    // Generate rotation samples with current sparsity
    const samples = generateRotationSamples(symClass, pointSparsityStep);
    const positions = [];
    const colors = [];

    // Map samples to 3D space and color by parameter value
    for (const sample of samples) {
      const { alpha, beta, gamma } = sample;
      const params = symAwareRotation(alpha, beta, gamma, symClass);

      // Position is the Euler angles (x=alpha, y=beta, z=gamma)
      positions.push(alpha, beta, gamma);

      // Update data bounds
      this.dataBounds.min.x = Math.min(this.dataBounds.min.x, alpha);
      this.dataBounds.min.y = Math.min(this.dataBounds.min.y, beta);
      this.dataBounds.min.z = Math.min(this.dataBounds.min.z, gamma);
      this.dataBounds.max.x = Math.max(this.dataBounds.max.x, alpha);
      this.dataBounds.max.y = Math.max(this.dataBounds.max.y, beta);
      this.dataBounds.max.z = Math.max(this.dataBounds.max.z, gamma);

      // color by the selected parameter
      let paramValue = 0;
      switch (this.paramName) {
        case 's_alpha': paramValue = params.s_a; break;
        case 'c_alpha': paramValue = params.c_a; break;
        case 's_beta': paramValue = params.s_b; break;
        case 'c_beta': paramValue = params.c_b; break;
        case 's_gamma': paramValue = params.s_g; break;
        case 'c_gamma': paramValue = params.c_g; break;
      }

      const color = viridiscolor(paramValue);
      colors.push(color.r, color.g, color.b);
    }

    // Create point cloud
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    function createCircleTexture(size = 64) {
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;

      const ctx = canvas.getContext('2d');
      const r = size / 2;

      const gradient = ctx.createRadialGradient(r, r, 0, r, r, r);
      gradient.addColorStop(0.0, 'rgba(255,255,255,1)');
      gradient.addColorStop(0.8, 'rgba(255,255,255,1)');
      gradient.addColorStop(1.0, 'rgba(255,255,255,0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(r, r, r, 0, Math.PI * 2);
      ctx.fill();

      return new THREE.CanvasTexture(canvas);
    }

    const material = new THREE.PointsMaterial({
      size: 0.15,
      map: createCircleTexture(),
      transparent: true,
      alphaTest: 0.3,
      depthWrite: false,
      vertexColors: true,
      sizeAttenuation: true
    });

    //const material = new THREE.PointsMaterial({
    //  size: 0.08,
    //  vertexColors: true,
    //  sizeAttenuation: true
    //});

    this.pointCloud = new THREE.Points(geometry, material);
    this.pointCloud.visible = showPointCloud;
    this.scene.add(this.pointCloud);

    // Add object primitives at selected positions (matching Python code sampling)
    // Generate separate samples for objects at fixed 5° sparsity (independent of point sparsity)
    const objectSamples = generateRotationSamples(symClass, 10);
    const objSet1Samples = objectSamples.filter(s => s.set === 1);
    const objSet2Samples = objectSamples.filter(s => s.set === 2);

    // Calculate sampling step based on objectSparsity settings
    // In SO(3) mode, use separate X, Y, Z densities; otherwise use Z sparsity for all
    let alphaModulo, betaModulo, gammaModulo;

    if (BOPSO3) {
      // Full SO(3): Use separate sparsity controls for each axis
      // Alpha (X-axis): 72 samples (360/5 = 72)
      alphaModulo = objectSparsityX;
      // Beta (Y-axis): 37 samples (180/5 + 1 = 37)
      betaModulo = objectSparsityY;
      // Gamma (Z-axis): 72 samples
      gammaModulo = objectSparsityZ;
    } else {
      // TLESS mode: Only Z-axis sparsity matters (beta is always 0)
      alphaModulo = objectSparsityX;
      betaModulo = objectSparsityY;  // Not used in TLESS mode
      gammaModulo = objectSparsityZ; //Math.max(1, Math.floor(360 / objectSparsityZ));
    }
    const safeModulo = (m) => (m <= 0 ? 1 : m);

  alphaModulo = safeModulo(alphaModulo);
  betaModulo  = safeModulo(betaModulo);
  gammaModulo = safeModulo(gammaModulo);

  // Helper to process one sample set
  const processSampleSet = (samples) => {
    const gammaValues = [...new Set(samples.map(s => s.gamma))];

    for (let gIdx = 0; gIdx < gammaValues.length; gIdx++) {
      if ((gIdx % gammaModulo) !== 0 && gIdx !== (gammaValues.length - 1)) {
        continue;
      }

      const gamma = gammaValues[gIdx];
      const samplesAtGamma = samples.filter(
        s => Math.abs(s.gamma - gamma) < 0.001
      );

      if (BOPSO3) {
        const betaValues = [...new Set(samplesAtGamma.map(s => s.beta))];

        for (let bIdx = 0; bIdx < betaValues.length; bIdx++) {
          if ((bIdx % betaModulo) !== 0 && bIdx !== (betaValues.length - 1)) {
            continue;
          }

          const beta = betaValues[bIdx];
          const samplesAtGammaBeta = samplesAtGamma.filter(
            s => Math.abs(s.beta - beta) < 0.001
          );

          for (let aIdx = 0; aIdx < samplesAtGammaBeta.length; aIdx++) {
            if ((aIdx % alphaModulo) !== 0) continue;

            const sample = samplesAtGammaBeta[aIdx];
            const primitive = createObjectPrimitive(symClass);

            primitive.rotation.order = 'XYZ';
            primitive.rotation.set(sample.alpha, sample.beta, sample.gamma);
            primitive.position.set(sample.alpha, sample.beta, sample.gamma);
            primitive.visible = showObjects;

            this.scene.add(primitive);
            this.objectPrimitives.push(primitive);
          }
        }
      } else {
        for (let aIdx = 0; aIdx < samplesAtGamma.length; aIdx++) {
          if ((aIdx % alphaModulo) !== 0) continue;

          const sample = samplesAtGamma[aIdx];
          const primitive = createObjectPrimitive(symClass);

          primitive.rotation.order = 'XYZ';
          primitive.rotation.set(sample.alpha, sample.beta, sample.gamma);
          primitive.position.set(sample.alpha, sample.beta, sample.gamma);
          primitive.visible = showObjects;

          this.scene.add(primitive);
          this.objectPrimitives.push(primitive);
        }
      }
    }
  };
    // Process both sets
    processSampleSet(objSet1Samples);
    processSampleSet(objSet2Samples);

}

  setPointCloudVisibility(visible) {
    if (this.pointCloud) {
      this.pointCloud.visible = visible;
    }
  }

  setGridVisibility(visible) {
    if (this.grid) {
      this.grid.visible = visible;
    }
  }

  setObjectsVisibility(visible) {
    this.objectPrimitives.forEach(primitive => {
      primitive.visible = visible;
    });
  }

  setAxesVisibility(visible) {
    if (this.parameterAxes) {
      this.parameterAxes.visible = visible;
    }
    // Trigger label update to show/hide them
    this.updateAxisLabels();
  }

  updateParameterSpace() {
    // Update bounds based on SO(3) mode
      // Full SO(3): α ∈ [0, 2π), β ∈ [0, 2π), γ ∈ [0, 2π) (matching Python)
    this.bounds.min.set(0, 0, 0);
    this.bounds.max.set(2 * Math.PI, 2 * Math.PI, 2 * Math.PI);


    // Remove old grid
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.geometry.dispose();
      this.grid.material.dispose();
    }

    // Create new grid based on bounds
    const centerX = (this.bounds.min.x + this.bounds.max.x) / 2;
    const centerY = (this.bounds.min.y + this.bounds.max.y) / 2;
    const centerZ = (this.bounds.min.z + this.bounds.max.z) / 2;

    const sizeX = this.bounds.max.x - this.bounds.min.x;
    const sizeZ = this.bounds.max.z - this.bounds.min.z;
    const gridDivisions = 20;

    this.grid = new THREE.GridHelper(Math.max(sizeX, sizeZ), gridDivisions, 0xaaaaaa, 0xcccccc);
    this.grid.rotation.x = Math.PI / 2;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.4;
    this.grid.position.set(centerX, centerY, centerZ);
    this.grid.visible = showGrid;
    this.scene.add(this.grid);

    // Update camera and controls target
    const distance = 8;
    this.camera.position.set(centerX + distance, centerY + distance, centerZ + distance);
    this.controls.target.set(centerX, centerY, centerZ);
    this.controls.update();

    // Update orbit center for auto-orbit feature
    this.orbitCenter.set(centerX, centerY, centerZ);
    this.orbitRadius = distance;

    if (this.parameterAxes) {
      this.scene.remove(this.parameterAxes);
      this.parameterAxes.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (child.material.map) child.material.map.dispose();
          child.material.dispose();
        }
      });
    }
    if (this.labelElements) {
      this.labelElements.forEach(el => el.remove());
      this.labelElements = null;
    }
    this.parameterAxes = this.createParameterAxes();
    this.parameterAxes.visible = showAxes;
    this.scene.add(this.parameterAxes);
  }
  resize() {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }
  render() {
    if (this.autoOrbit) {
      this.orbitAngle += this.orbitSpeed;
      const angleRad = this.orbitAngle * Math.PI / 180;
      const x = this.orbitCenter.x + this.orbitRadius * Math.cos(angleRad);
      const z = this.orbitCenter.z + this.orbitRadius * Math.sin(angleRad);
      const y = this.orbitCenter.y + this.orbitRadius * 0.5; // Slight elevation

      this.camera.position.set(x, y, z);
      this.camera.lookAt(this.orbitCenter);
      this.controls.target.copy(this.orbitCenter);
      this.controls.update();
    } else {
      this.controls.update();
    }

    this.renderer.render(this.scene, this.camera);
    this.updateAxisLabels();
  }

  updateAxisLabels() {
    // Update DOM-based LaTeX labels positions
    if (!this.axisLabels || !this.parameterAxes || !this.parameterAxes.visible) {
      // Hide labels if axes are not visible
      if (this.labelElements) {
        this.labelElements.forEach(el => el.style.display = 'none');
      }
      return;
    }

    // Create label elements if they don't exist
    if (!this.labelElements) {
      this.labelElements = [];
      const colorHex = '#' + axisColor.toString(16).padStart(6, '0');

      this.axisLabels.forEach((label, index) => {
        const div = document.createElement('div');
        div.className = 'axis-label';
        div.style.position = 'absolute';
        div.style.color = colorHex;
        div.style.fontSize = '24px';
        div.style.fontWeight = 'bold';
        div.style.pointerEvents = 'none';
        div.style.userSelect = 'none';
        div.style.zIndex = '1000';
        div.innerHTML = label.text;
        this.container.appendChild(div);
        this.labelElements.push(div);
      });

      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise(this.labelElements).catch(err => console.log('MathJax error:', err));
      }
    }

    this.axisLabels.forEach((label, index) => {
      const screenPos = this.toScreenPosition(label.position);
      const element = this.labelElements[index];
      element.style.display = 'block';
      element.style.left = screenPos.x + 'px';
      element.style.top = screenPos.y + 'px';
    });
  }
  toScreenPosition(position) {
    const vector = position.clone();
    vector.project(this.camera);
    const widthHalf = this.container.clientWidth / 2;
    const heightHalf = this.container.clientHeight / 2;
    return {
      x: (vector.x * widthHalf) + widthHalf,
      y: -(vector.y * heightHalf) + heightHalf
    };
  }
  setAutoOrbit(enabled) {
    this.autoOrbit = enabled;
    if (enabled) {
      const dx = this.camera.position.x - this.orbitCenter.x;
      const dz = this.camera.position.z - this.orbitCenter.z;
      this.orbitAngle = Math.atan2(dz, dx) * 180 / Math.PI;
      this.controls.enabled = false; // Disable manual controls during auto-orbit
    } else {
      this.controls.enabled = true; // Re-enable manual controls
    }
  }
}
function updateUIFromConfig() {
  if (!config || !config.visualization) return;

  // Update sliders with config values
  if (sparsitySlider && config.visualization.pointSparsity) {
    const cfg = config.visualization.pointSparsity;
    sparsitySlider.min = cfg.min;
    sparsitySlider.max = cfg.max;
    sparsitySlider.step = cfg.step;
    sparsitySlider.value = pointSparsityStep;
    if (sparsityValue) sparsityValue.textContent = `${pointSparsityStep}${cfg.unit}`;
  }

  if (objectSparsityXSlider && config.visualization.objectSparsityX) {
    const cfg = config.visualization.objectSparsityX;
    objectSparsityXSlider.min = cfg.min;
    objectSparsityXSlider.max = cfg.max;
    objectSparsityXSlider.step = cfg.step;
    objectSparsityXSlider.value = objectSparsityX;
    if (objectSparsityXValue) objectSparsityXValue.textContent = `${objectSparsityX}${cfg.unit}`;
  }

  if (objectSparsityYSlider && config.visualization.objectSparsityY) {
    const cfg = config.visualization.objectSparsityY;
    objectSparsityYSlider.min = cfg.min;
    objectSparsityYSlider.max = cfg.max;
    objectSparsityYSlider.step = cfg.step;
    objectSparsityYSlider.value = objectSparsityY;
    if (objectSparsityYValue) objectSparsityYValue.textContent = `${objectSparsityY}${cfg.unit}`;
  }

    if (objectSparsityZSlider && config.visualization.objectSparsityZ) {
    const cfg = config.visualization.objectSparsityZ;
    objectSparsityZSlider.min = cfg.min;
    objectSparsityZSlider.max = cfg.max;
    objectSparsityZSlider.step = cfg.step;
    objectSparsityZSlider.value = objectSparsityZ;
    if (objectSparsityZValue) objectSparsityZValue.textContent = `${objectSparsityZ}${cfg.unit}`;
  }

  // Update toggle states
  if (toggleBOPSO3) toggleBOPSO3.checked = BOPSO3;
  if (togglePoints) togglePoints.checked = showPointCloud;
  if (toggleGrid) toggleGrid.checked = showGrid;
  if (toggleAxes) toggleAxes.checked = showAxes;
  if (toggleObjects) toggleObjects.checked = showObjects;

  if (BOPSO3 === false){

  }
}
const sparsitySlider = document.getElementById('sparsitySlider');
const sparsityValue = document.getElementById('sparsityValue');
const objectSparsityXSlider = document.getElementById('objectSparsityXSlider');
const objectSparsityXValue = document.getElementById('objectSparsityXValue');
const objectSparsityYSlider = document.getElementById('objectSparsityYSlider');
const objectSparsityYValue = document.getElementById('objectSparsityYValue');
const objectSparsityZSlider = document.getElementById('objectSparsityZSlider');
const objectSparsityZValue = document.getElementById('objectSparsityZValue');
const toggleBOPSO3 = document.getElementById('toggleBOPSO3');
const togglePoints = document.getElementById('togglePoints');
const toggleGrid = document.getElementById('toggleGrid');
const toggleAxes = document.getElementById('toggleAxes');
const toggleObjects = document.getElementById('toggleObjects');

if (sparsitySlider) {
  pointSparsityStep = parseInt(sparsitySlider.value);
  if (sparsityValue) sparsityValue.textContent = `${pointSparsityStep}°`;
}
if (objectSparsityZSlider) {
  objectSparsityZ = parseInt(objectSparsityZSlider.value);
  if (objectSparsityZValue) objectSparsityZValue.textContent = `${objectSparsityZ}`;
}
if (objectSparsityXSlider) {
  objectSparsityX = parseInt(objectSparsityXSlider.value);
  if (objectSparsityXValue) objectSparsityXValue.textContent = `${objectSparsityX}`;
}
if (objectSparsityYSlider) {
  objectSparsityY = parseInt(objectSparsityYSlider.value);
  if (objectSparsityYValue) objectSparsityYValue.textContent = `${objectSparsityY}`;
}
if (toggleBOPSO3) BOPSO3 = toggleBOPSO3.checked;
if (togglePoints) showPointCloud = togglePoints.checked;
if (toggleGrid) showGrid = toggleGrid.checked;
if (toggleAxes) showAxes = toggleAxes.checked;
if (toggleObjects) showObjects = toggleObjects.checked;

if (BOPSO3 === false){
  sparsitySlider.disabled = true;
  objectSparsityXSlider.disabled = true;
  objectSparsityYSlider.disabled = true;
  objectSparsityZSlider.disabled = true;
}
let currentSymmetryClass = '1';
const viewers = [];
const paramNames = ['s_alpha', 's_beta', 's_gamma', 'c_alpha', 'c_beta', 'c_gamma'];



for (let i = 0; i < 6; i++) {
  const viewer = new SubplotViewer(`viewer-${i}`, paramNames[i], i);
  viewers.push(viewer);
  viewer.updateVisualization(currentSymmetryClass);
}
viewers.forEach(viewer => {
  viewer.setPointCloudVisibility(showPointCloud);
  viewer.setGridVisibility(showGrid);
  viewer.setAxesVisibility(showAxes);
  viewer.setObjectsVisibility(showObjects);
});
function animate() {
  requestAnimationFrame(animate);
  viewers.forEach(viewer => viewer.render());
  insetViewer.render();
}
animate();
function onResize() {
  viewers.forEach(viewer => viewer.resize());
}
window.addEventListener('resize', onResize);

const subplotGrid = document.querySelector('.subplot-grid');
const maximizeButtons = document.querySelectorAll('.btn-maximize');
maximizeButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    const viewerIdx = parseInt(btn.dataset.viewer);
    const container = btn.closest('.subplot-container');
    const isMaximized = container.classList.contains('maximized');

    if (isMaximized) {
      container.classList.remove('maximized');
      subplotGrid.classList.remove('has-maximized');
      btn.textContent = '\u26F6';
      btn.title = 'Maximize panel';
      insetViewer.hide();
    } else {
      document.querySelectorAll('.subplot-container.maximized').forEach(el => {
        el.classList.remove('maximized');
        el.querySelector('.btn-maximize').textContent = '\u26F6';
        el.querySelector('.btn-maximize').title = 'Maximize panel';
      });
      container.classList.add('maximized');
      subplotGrid.classList.add('has-maximized');
      btn.textContent = '\u2716';
      btn.title = 'Restore panel';
      insetViewer.show(container, currentSymmetryClass);
    }

    setTimeout(() => {
      viewers.forEach(viewer => viewer.resize());
    }, 50);
  });
});
const symButtons = document.querySelectorAll('.btn-sym');
symButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    symButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSymmetryClass = btn.dataset.class;
    viewers.forEach(viewer => viewer.updateVisualization(currentSymmetryClass));
    const maximizedPanel = document.querySelector('.subplot-container.maximized');
    if (maximizedPanel) {
      insetViewer.show(maximizedPanel, currentSymmetryClass);
    }
  });
});
if (sparsitySlider && sparsityValue) {
  sparsitySlider.addEventListener('input', () => {
    pointSparsityStep = parseInt(sparsitySlider.value);
    const unit = config?.visualization?.pointSparsity?.unit || '°';
    sparsityValue.textContent = `${pointSparsityStep}${unit}`;
    // Update all viewers
    viewers.forEach(viewer => viewer.updateVisualization(currentSymmetryClass));
  });
}
if (objectSparsityZSlider && objectSparsityZValue) {
  objectSparsityZSlider.addEventListener('input', () => {
    objectSparsityZ = parseInt(objectSparsityZSlider.value);
    objectSparsityZValue.textContent = `${objectSparsityZ}`;
    // Update all viewers
    viewers.forEach(viewer => viewer.updateVisualization(currentSymmetryClass));
  });
}
if (objectSparsityXSlider && objectSparsityXValue) {
  objectSparsityXSlider.addEventListener('input', () => {
    objectSparsityX = parseInt(objectSparsityXSlider.value);
    objectSparsityXValue.textContent = `${objectSparsityX}`;
    // Update all viewers
    viewers.forEach(viewer => viewer.updateVisualization(currentSymmetryClass));
  });
}
if (objectSparsityYSlider && objectSparsityYValue) {
  objectSparsityYSlider.addEventListener('input', () => {
    objectSparsityY = parseInt(objectSparsityYSlider.value);
    objectSparsityYValue.textContent = `${objectSparsityY}`;
    // Update all viewers
    viewers.forEach(viewer => viewer.updateVisualization(currentSymmetryClass));
  });
}
if (toggleBOPSO3) {
  toggleBOPSO3.addEventListener('change', () => {
    BOPSO3 = toggleBOPSO3.checked;
    if (BOPSO3) {
      const autoAdjust = config?.SO3AutoAdjust || {
        pointSparsity: 10,
        objectSparsityZ: 8,
        objectSparsityX: 8,
        objectSparsityY: 8
      };
      pointSparsityStep = autoAdjust.pointSparsity;
      objectSparsityZ = autoAdjust.objectSparsityZ;
      objectSparsityX = autoAdjust.objectSparsityX;
      objectSparsityY = autoAdjust.objectSparsityY;
      if (sparsitySlider) {
        sparsitySlider.value = pointSparsityStep;
        sparsitySlider.disabled = false;
        if (sparsityValue) {
          const unit = config?.visualization?.pointSparsity?.unit || '°';
          sparsityValue.textContent = `${pointSparsityStep}${unit}`;
        }
      }
      if (objectSparsityXSlider) {
        objectSparsityXSlider.value = objectSparsityX;
        objectSparsityXSlider.disabled = false; // Enable slider
        if (objectSparsityXValue) objectSparsityXValue.textContent = `${objectSparsityX}`;
      }
      if (objectSparsityYSlider) {
        objectSparsityYSlider.value = objectSparsityY;
        objectSparsityYSlider.disabled = false; // Enable slider
        if (objectSparsityYValue) objectSparsityYValue.textContent = `${objectSparsityY}`;
      }
      if (objectSparsityZSlider) {
        objectSparsityZSlider.value = objectSparsityZ;
        objectSparsityZSlider.disabled = false; // Enable slider
        if (objectSparsityZValue) objectSparsityZValue.textContent = `${objectSparsityZ}`;
      }
    } else {
        const autoAdjust = config?.BOPAutoAdjust || {
        pointSparsity: 10,
        objectSparsityX: 2,
        objectSparsityY: -1,
        objectSparsityZ: 6,
        };
      pointSparsityStep = autoAdjust.pointSparsity;
      objectSparsityZ = autoAdjust.objectSparsityZ;
      objectSparsityX = autoAdjust.objectSparsityX;
      objectSparsityY = autoAdjust.objectSparsityY;
      if (sparsitySlider) {
        sparsitySlider.value = pointSparsityStep;
        if (sparsityValue) {
          const unit = config?.visualization?.pointSparsity?.unit || '°';
          sparsityValue.textContent = `${pointSparsityStep}${unit}`;
        }
        sparsitySlider.disabled = true;
      }
      if (objectSparsityXSlider) {
        objectSparsityXSlider.value = objectSparsityX;
        objectSparsityXSlider.disabled = true;
        if (objectSparsityXValue) objectSparsityXValue.textContent = `${objectSparsityX}`;
      }
      if (objectSparsityYSlider) {
        objectSparsityYSlider.value = objectSparsityY;
        objectSparsityYSlider.disabled = true;
        if (objectSparsityYValue) objectSparsityYValue.textContent = `${objectSparsityY}`;
      }
      if (objectSparsityZSlider) {
        objectSparsityZSlider.value = objectSparsityZ;
        objectSparsityZSlider.disabled = true;
        if (objectSparsityZValue) objectSparsityZValue.textContent = `${objectSparsityZ}`;
      }
    }
    viewers.forEach(viewer => {
      viewer.updateParameterSpace();
      viewer.updateVisualization(currentSymmetryClass);
    });
  });
}
if (togglePoints) {
  togglePoints.addEventListener('change', () => {
    showPointCloud = togglePoints.checked;
    viewers.forEach(viewer => viewer.setPointCloudVisibility(showPointCloud));
  });
}
if (toggleGrid) {
  toggleGrid.addEventListener('change', () => {
    showGrid = toggleGrid.checked;
    viewers.forEach(viewer => viewer.setGridVisibility(showGrid));
  });
}
if (toggleAxes) {
  toggleAxes.addEventListener('change', () => {
    showAxes = toggleAxes.checked;
    viewers.forEach(viewer => viewer.setAxesVisibility(showAxes));
  });
}
if (toggleObjects) {
  toggleObjects.addEventListener('change', () => {
    showObjects = toggleObjects.checked;
    viewers.forEach(viewer => viewer.setObjectsVisibility(showObjects));
  });
}
// Color palette selector handler
const colorPaletteSelect = document.getElementById('colorPaletteSelect');
if (colorPaletteSelect) {
  colorPaletteSelect.addEventListener('change', () => {
    colorPalette = colorPaletteSelect.value;
    // Regenerate all visualizations with new colors
    viewers.forEach(viewer => viewer.updateVisualization(currentSymmetryClass));
  });
}
const btnOrbitAll = document.getElementById('btnOrbitAll');
if (btnOrbitAll) {
  let isOrbiting = false;
  btnOrbitAll.addEventListener('click', () => {
    isOrbiting = !isOrbiting;
    if (isOrbiting) {
      btnOrbitAll.classList.add('active');
    } else {
      btnOrbitAll.classList.remove('active');
    }
    viewers.forEach(viewer => viewer.setAutoOrbit(isOrbiting));
  });
}
const btnMinimizeControls = document.getElementById('btnMinimizeControls');
const controlsPanel = document.getElementById('controlsPanel');
if (btnMinimizeControls && controlsPanel) {
  btnMinimizeControls.addEventListener('click', () => {
    const isMinimized = controlsPanel.classList.toggle('minimized');
    btnMinimizeControls.textContent = isMinimized ? '+' : '−';
    btnMinimizeControls.title = isMinimized ? 'Maximize panel' : 'Minimize panel';
  });
}
const themeSelect = document.getElementById('themeSelect');
if (themeSelect) {
  const storedTheme = getGlobalSetting('theme', 'gray');
  themeSelect.value = storedTheme;
  document.documentElement.className = 'theme-' + storedTheme;
  themeSelect.addEventListener('change', () => {
    const theme = themeSelect.value;
    document.documentElement.className = 'theme-' + theme;
    setGlobalSetting('theme', theme);
  });
}