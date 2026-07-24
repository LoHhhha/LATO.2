import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

const SINGLE_MESH_DIRECTORY = "assets/meshes/single/";
const SINGLE_MESH_MANIFEST = `${SINGLE_MESH_DIRECTORY}manifest.json`;
const CONTROL_MESH_DIRECTORY = "assets/meshes/control/";
const CONTROL_MESH_MANIFEST = `${CONTROL_MESH_DIRECTORY}manifest.json`;
const MULTI_MESH_DIRECTORY = "assets/meshes/multi/";
const MULTI_MESH_MANIFEST = `${MULTI_MESH_DIRECTORY}manifest.json`;
const EDIT_MESH_DIRECTORY = "assets/meshes/edit/";
const EDIT_MESH_MANIFEST = `${EDIT_MESH_DIRECTORY}manifest.json`;
const CONTROL_BUDGETS = [200, 500, 1000, 2000, 3000, 4000];
const MESHES_PER_SET = 3;
const EDIT_POINT_SIZE_SCALE = 0.036;
const PART_EXPLOSION_DISTANCE_SCALE = 0.62;
const PART_EXPLOSION_CAMERA_SCALE = 0.58;
const formatIndex = (value) => String(value).padStart(2, "0");
const formatCount = (value) => value.toLocaleString("en-US");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const naturalSort = (first, second) => first.localeCompare(second, undefined, {
  numeric: true,
  sensitivity: "base",
});

const groupItems = (items, groupSize) => {
  const groups = [];

  for (let index = 0; index < items.length; index += groupSize) {
    groups.push(items.slice(index, index + groupSize));
  }

  return groups;
};

const createSingleMeshItems = (fileNames) => fileNames
  .filter((fileName) => typeof fileName === "string" && fileName.toLowerCase().endsWith(".obj"))
  .sort(naturalSort)
  .map((fileName, index) => ({
    fileName,
    path: `${SINGLE_MESH_DIRECTORY}${encodeURIComponent(fileName)}`,
    label: `Generated Mesh ${formatIndex(index + 1)}`,
  }));

const loadSingleMeshItems = async (group) => {
  try {
    const response = await fetch(SINGLE_MESH_MANIFEST, { cache: "no-store" });
    if (!response.ok) throw new Error(`Manifest request failed with ${response.status}`);

    const manifest = await response.json();
    const items = createSingleMeshItems(manifest.meshes || []);
    if (items.length) return items;
  } catch (error) {
    console.warn("Unable to load the single-generation mesh manifest; using the HTML fallback.", error);
  }

  return [...group.querySelectorAll("[data-obj-path]")]
    .map((container, index) => {
      const path = container.dataset.objPath;
      if (!path) return null;

      return {
        fileName: decodeURIComponent(path.split("/").at(-1)),
        path,
        label: `Generated Mesh ${formatIndex(index + 1)}`,
      };
    })
    .filter(Boolean);
};

const normalizeControlSet = (set) => {
  const meshesByBudget = new Map(
    (set.meshes || []).map((mesh) => [Number(mesh.budget), mesh.file])
  );

  return {
    name: set.name || "",
    items: CONTROL_BUDGETS.map((budget) => {
      const fileName = meshesByBudget.get(budget);
      if (!fileName) return null;

      return {
        budget,
        fileName,
        path: `${CONTROL_MESH_DIRECTORY}${encodeURIComponent(fileName)}`,
        label: `${formatCount(budget)} Vertices`,
      };
    }),
  };
};

const loadControlSets = async (group) => {
  try {
    const response = await fetch(CONTROL_MESH_MANIFEST, { cache: "no-store" });
    if (!response.ok) throw new Error(`Manifest request failed with ${response.status}`);

    const manifest = await response.json();
    const sets = (manifest.sets || [])
      .map(normalizeControlSet)
      .filter((set) => set.items.some(Boolean));
    if (sets.length) return sets;
  } catch (error) {
    console.warn("Unable to load the mesh-complexity manifest; using the HTML fallback.", error);
  }

  const fallbackSet = {
    name: "",
    items: CONTROL_BUDGETS.map((budget) => {
      const container = group.querySelector(`[data-budget="${budget}"]`);
      const path = container?.dataset.objPath;
      if (!path) return null;

      return {
        budget,
        fileName: decodeURIComponent(path.split("/").at(-1)),
        path,
        label: `${formatCount(budget)} Vertices`,
      };
    }),
  };

  return fallbackSet.items.some(Boolean) ? [fallbackSet] : [];
};

const createMultiMeshItems = (fileNames) => fileNames
  .filter((fileName) => typeof fileName === "string" && fileName.toLowerCase().endsWith(".glb"))
  .sort(naturalSort)
  .map((fileName, index) => ({
    fileName,
    path: `${MULTI_MESH_DIRECTORY}${encodeURIComponent(fileName)}`,
    label: `Part-wise Generation Result ${formatIndex(index + 1)}`,
  }));

const loadMultiMeshItems = async (group) => {
  try {
    const response = await fetch(MULTI_MESH_MANIFEST, { cache: "no-store" });
    if (!response.ok) throw new Error(`Manifest request failed with ${response.status}`);

    const manifest = await response.json();
    const items = createMultiMeshItems(manifest.meshes || []);
    if (items.length) return items;
  } catch (error) {
    console.warn("Unable to load the part-wise GLB manifest; using the HTML fallback.", error);
  }

  return [...group.querySelectorAll("[data-glb-path]")]
    .map((container, index) => {
      const path = container.dataset.glbPath;
      if (!path) return null;

      return {
        fileName: decodeURIComponent(path.split("/").at(-1)),
        path,
        label: `Part-wise Generation Result ${formatIndex(index + 1)}`,
      };
    })
    .filter(Boolean);
};

const normalizeEditSet = (set) => ({
  name: typeof set.name === "string" ? set.name : "",
  items: (set.meshes || [])
    .slice(0, 4)
    .map((mesh) => {
      const fileName = typeof mesh.file === "string" ? mesh.file : "";
      const type = ["vertex", "pointcloud"].includes(mesh.type) ? "vertex" : "mesh";
      if (!fileName || ![".obj", ".ply"].some((extension) => fileName.toLowerCase().endsWith(extension))) {
        return null;
      }

      return {
        name: typeof mesh.name === "string" && mesh.name.trim() ? mesh.name.trim() : fileName,
        type,
        fileName,
        path: `${EDIT_MESH_DIRECTORY}${encodeURIComponent(fileName)}`,
      };
    }),
});

const loadEditSets = async (group) => {
  try {
    const response = await fetch(EDIT_MESH_MANIFEST, { cache: "no-store" });
    if (!response.ok) throw new Error(`Manifest request failed with ${response.status}`);

    const manifest = await response.json();
    const sets = (manifest.sets || [])
      .map(normalizeEditSet)
      .filter((set) => set.items.some(Boolean));
    if (sets.length) return sets;
  } catch (error) {
    console.warn("Unable to load the editing manifest; using the HTML fallback.", error);
  }

  const items = [...group.querySelectorAll("[data-edit-path]")]
    .slice(0, 4)
    .map((container) => {
      const path = container.dataset.editPath;
      if (!path) return null;

      return {
        name: container.dataset.editName || decodeURIComponent(path.split("/").at(-1)),
        type: ["vertex", "pointcloud"].includes(container.dataset.editType) ? "vertex" : "mesh",
        fileName: decodeURIComponent(path.split("/").at(-1)),
        path,
      };
    });

  return items.some(Boolean) ? [{ name: "", items }] : [];
};

const readPalette = () => {
  const styles = getComputedStyle(document.documentElement);
  return {
    meshBottom: [0.012, 0.045, 0.20],
    meshTop: [0.9, 0.4, 0.32],
    accentDark: styles.getPropertyValue("--accent-dark").trim() || "#4f3527",
  };
};

const parseObjStats = (source) => {
  let vertices = 0;
  let faces = 0;

  source.split(/\r?\n/).forEach((line) => {
    const normalized = line.trimStart();

    if (normalized.startsWith("v ")) {
      vertices += 1;
      return;
    }

    if (normalized.startsWith("f ")) {
      const vertexReferences = normalized.trim().split(/\s+/).length - 1;
      faces += Math.max(vertexReferences - 2, 0);
    }
  });

  return { vertices, faces };
};

const disposeMaterial = (material) => {
  if (Array.isArray(material)) {
    material.forEach(disposeMaterial);
    return;
  }

  material?.dispose();
};

const disposeObject = (object) => {
  object.traverse((child) => {
    child.geometry?.dispose();
    disposeMaterial(child.material);
  });
};

const prepareGlbScene = (object, palette, viewport) => {
  let vertices = 0;
  let faces = 0;
  const meshes = [];
  const wireColor = new THREE.Color(palette.accentDark);
  const wireMaterials = [];

  object.traverse((child) => {
    const geometry = child.geometry;
    const positions = geometry?.attributes?.position;
    if (!child.isMesh || !positions) return;

    meshes.push(child);
  });

  meshes.forEach((child) => {
    const geometry = child.geometry;
    const positions = geometry.attributes.position;

    vertices += positions.count;
    faces += geometry.index ? geometry.index.count / 3 : positions.count / 3;

    if (!geometry.attributes.normal) geometry.computeVertexNormals();

    const hasVertexColors = Boolean(geometry.attributes.color);
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => {
      if (!material) return;

      material.vertexColors = hasVertexColors;
      material.side = THREE.DoubleSide;
      if ("roughness" in material) material.roughness = 0.8;
      if ("metalness" in material) material.metalness = 0.02;
      material.needsUpdate = true;
    });

    const sourceWireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeGeometry = new LineSegmentsGeometry()
      .fromWireframeGeometry(sourceWireframeGeometry);
    const wireframeMaterial = new LineMaterial({
      color: wireColor,
      transparent: true,
      opacity: 0.3,
      linewidth: 1.35,
      alphaToCoverage: true,
      depthWrite: false,
    });
    const wireframe = new LineSegments2(
      wireframeGeometry,
      wireframeMaterial
    );

    sourceWireframeGeometry.dispose();
    wireframeMaterial.resolution.set(viewport.width, viewport.height);
    wireframe.renderOrder = 2;
    child.add(wireframe);
    wireMaterials.push(wireframeMaterial);
  });

  return {
    vertices,
    faces: Math.floor(faces),
    parts: meshes.length,
    meshes,
    wireMaterials,
  };
};

const createExplosionParts = (meshes, radius) => meshes.map((mesh, index) => {
  const geometry = mesh.geometry;
  if (!geometry.boundingBox) geometry.computeBoundingBox();

  const localCenter = geometry.boundingBox.getCenter(new THREE.Vector3());
  const worldCenter = mesh.localToWorld(localCenter.clone());
  const worldDirection = worldCenter.clone();

  if (worldDirection.length() < radius * 0.03) {
    const angle = index * Math.PI * (3 - Math.sqrt(5));
    const vertical = ((index % 5) - 2) * 0.18;
    worldDirection.set(Math.cos(angle), vertical, Math.sin(angle));
  }
  worldDirection.normalize();

  const parent = mesh.parent;
  const localOrigin = parent.worldToLocal(new THREE.Vector3());
  const localEnd = parent.worldToLocal(worldDirection.clone());
  const direction = localEnd.sub(localOrigin).normalize();

  return {
    mesh,
    basePosition: mesh.position.clone(),
    direction,
  };
});

const prepareColoredObj = (object, palette, viewport) => {
  const wireColor = new THREE.Color(palette.accentDark);
  const wireMaterials = [];
  const meshes = [];

  object.traverse((child) => {
    const geometry = child.geometry;
    const positions = geometry?.attributes?.position;
    if (child.isMesh && positions) meshes.push(child);
  });

  meshes.forEach((child) => {
    const geometry = child.geometry;

    if (!geometry.attributes.normal) geometry.computeVertexNormals();
    const hasVertexColors = Boolean(geometry.attributes.color);

    disposeMaterial(child.material);
    child.material = new THREE.MeshStandardMaterial({
      color: hasVertexColors ? 0xffffff : 0xd8cec7,
      vertexColors: hasVertexColors,
      side: THREE.DoubleSide,
      roughness: 0.8,
      metalness: 0.02,
      flatShading: true,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });

    const sourceWireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeGeometry = new LineSegmentsGeometry()
      .fromWireframeGeometry(sourceWireframeGeometry);
    const wireframeMaterial = new LineMaterial({
      color: wireColor,
      transparent: true,
      opacity: 0.26,
      linewidth: 1.2,
      alphaToCoverage: true,
      depthWrite: false,
    });
    const wireframe = new LineSegments2(
      wireframeGeometry,
      wireframeMaterial
    );

    sourceWireframeGeometry.dispose();
    wireframeMaterial.resolution.set(viewport.width, viewport.height);
    wireframe.renderOrder = 2;
    child.add(wireframe);
    wireMaterials.push(wireframeMaterial);
  });

  return wireMaterials;
};

const createColoredPointCloud = (geometry) => {
  const hasVertexColors = Boolean(geometry.attributes.color);
  const material = new THREE.PointsMaterial({
    color: hasVertexColors ? 0xffffff : 0xb06a50,
    vertexColors: hasVertexColors,
    size: EDIT_POINT_SIZE_SCALE,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.96,
  });

  return new THREE.Points(geometry, material);
};

const colorizeMesh = (object, bounds, palette, viewport) => {
  const lowColor = new THREE.Color().setRGB(...palette.meshBottom);
  const highColor = new THREE.Color().setRGB(...palette.meshTop);
  const wireColor = new THREE.Color(palette.accentDark);
  const height = Math.max(bounds.max.y - bounds.min.y, Number.EPSILON);
  const wireMaterials = [];
  const meshes = [];

  object.traverse((child) => {
    if (child.isMesh && child.geometry?.attributes?.position) meshes.push(child);
  });

  meshes.forEach((child) => {
    const geometry = child.geometry;
    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);
    const color = new THREE.Color();

    for (let index = 0; index < positions.count; index += 1) {
      const normalizedHeight = THREE.MathUtils.clamp(
        (positions.getY(index) - bounds.min.y) / height,
        0,
        1
      );
      const blend = normalizedHeight * normalizedHeight * (3 - 2 * normalizedHeight);

      color.copy(lowColor).lerp(highColor, blend);
      colors[index * 3] = color.r;
      colors[index * 3 + 1] = color.g;
      colors[index * 3 + 2] = color.b;
    }

    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    disposeMaterial(child.material);
    child.material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.78,
      metalness: 0.04,
      flatShading: true,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });

    const sourceWireframeGeometry = new THREE.WireframeGeometry(geometry);
    const wireframeGeometry = new LineSegmentsGeometry()
      .fromWireframeGeometry(sourceWireframeGeometry);
    const wireframeMaterial = new LineMaterial({
      color: wireColor,
      transparent: true,
      opacity: 0.34,
      linewidth: 1.35,
      alphaToCoverage: true,
      depthWrite: false,
    });
    const wireframe = new LineSegments2(
      wireframeGeometry,
      wireframeMaterial
    );

    sourceWireframeGeometry.dispose();
    wireframeMaterial.resolution.set(viewport.width, viewport.height);
    wireframe.renderOrder = 2;
    child.add(wireframe);
    wireMaterials.push(wireframeMaterial);
  });

  return wireMaterials;
};

const createObjViewer = (container, slot) => {
  const palette = readPalette();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  const controls = new OrbitControls(camera, renderer.domElement);
  const loader = new OBJLoader();

  let currentObject = null;
  let currentRequest = 0;
  let wireMaterials = [];
  let visible = true;

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xf8f3ef, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.domElement.setAttribute("aria-hidden", "true");
  container.prepend(renderer.domElement);

  controls.enableDamping = true;
  controls.dampingFactor = 0.055;
  controls.enablePan = false;
  controls.autoRotate = !reducedMotion.matches;
  controls.autoRotateSpeed = 0.72;

  scene.add(new THREE.HemisphereLight(0xfff7f0, 0x667581, 2.1));

  const keyLight = new THREE.DirectionalLight(0xfff4e9, 3.2);
  keyLight.position.set(4, 6, 5);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0xb8cad4, 1.9);
  rimLight.position.set(-5, 2, -4);
  scene.add(rimLight);

  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    wireMaterials.forEach((material) => material.resolution.set(width, height));
  };

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  if ("IntersectionObserver" in window) {
    const visibilityObserver = new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting ?? true;
    }, { rootMargin: "180px 0px" });
    visibilityObserver.observe(container);
  }

  const animate = () => {
    requestAnimationFrame(animate);
    if (!visible) return;
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

  const update = async (item) => {
    const requestId = ++currentRequest;
    const loading = container.querySelector(".mesh-viewer__loading");
    const faces = slot.querySelector("[data-mesh-faces]");
    const vertices = slot.querySelector("[data-mesh-vertices]");
    const actualVertices = slot.querySelector("[data-actual-vertices]");

    container.classList.remove("is-ready", "is-error");
    loading.textContent = "Loading mesh";
    if (faces) faces.textContent = "—";
    if (vertices) vertices.textContent = "—";
    if (actualVertices) actualVertices.textContent = "—";

    try {
      const response = await fetch(item.path);
      if (!response.ok) throw new Error(`OBJ request failed with ${response.status}`);

      const source = await response.text();
      if (requestId !== currentRequest) return;

      const stats = parseObjStats(source);
      const object = loader.parse(source);
      const initialBounds = new THREE.Box3().setFromObject(object);
      const center = initialBounds.getCenter(new THREE.Vector3());

      object.position.sub(center);
      object.updateMatrixWorld(true);

      const centeredBounds = new THREE.Box3().setFromObject(object);
      const sphere = centeredBounds.getBoundingSphere(new THREE.Sphere());
      const radius = Math.max(sphere.radius, 0.01);

      const nextWireMaterials = colorizeMesh(object, centeredBounds, palette, {
        width: Math.max(container.clientWidth, 1),
        height: Math.max(container.clientHeight, 1),
      });

      if (currentObject) {
        scene.remove(currentObject);
        disposeObject(currentObject);
      }

      currentObject = object;
      wireMaterials = nextWireMaterials;
      scene.add(object);

      camera.near = Math.max(radius / 100, 0.001);
      camera.far = radius * 30;
      camera.position.set(radius * 2.55, radius * 1.25, radius * 2.85);
      camera.updateProjectionMatrix();

      controls.target.set(0, 0, 0);
      controls.minDistance = radius * 1.25;
      controls.maxDistance = radius * 8;
      controls.update();

      if (faces) faces.textContent = formatCount(stats.faces);
      if (vertices) vertices.textContent = formatCount(stats.vertices);
      if (actualVertices) actualVertices.textContent = formatCount(stats.vertices);
      container.dataset.objPath = item.path;
      container.setAttribute("aria-label", `Interactive viewer for ${item.label}`);
      container.classList.add("is-ready");
    } catch (error) {
      if (requestId !== currentRequest) return;
      console.error(`Unable to load ${item.path}`, error);
      loading.textContent = "Mesh could not be loaded";
      container.classList.add("is-error");
    }
  };

  reducedMotion.addEventListener("change", (event) => {
    controls.autoRotate = !event.matches;
  });

  return { update };
};

const createGlbViewer = (container, slot) => {
  const palette = readPalette();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  const controls = new OrbitControls(camera, renderer.domElement);
  const loader = new GLTFLoader();

  let currentObject = null;
  let currentRequest = 0;
  let wireMaterials = [];
  let explosionParts = [];
  let explosionDistance = 0;
  let currentExplosion = 0;
  let appliedCameraExplosion = 0;
  let cameraReady = false;
  let visible = true;

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xf8f3ef, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.04;
  renderer.domElement.setAttribute("aria-hidden", "true");
  container.prepend(renderer.domElement);

  controls.enableDamping = true;
  controls.dampingFactor = 0.055;
  controls.enablePan = false;
  controls.autoRotate = !reducedMotion.matches;
  controls.autoRotateSpeed = 0.72;

  scene.add(new THREE.HemisphereLight(0xfff7f0, 0x667581, 2.15));

  const keyLight = new THREE.DirectionalLight(0xfff4e9, 3.1);
  keyLight.position.set(4, 6, 5);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0xb8cad4, 1.85);
  rimLight.position.set(-5, 2, -4);
  scene.add(rimLight);

  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    wireMaterials.forEach((material) => material.resolution.set(width, height));
  };

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  if ("IntersectionObserver" in window) {
    const visibilityObserver = new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting ?? true;
    }, { rootMargin: "180px 0px" });
    visibilityObserver.observe(container);
  }

  const animate = () => {
    requestAnimationFrame(animate);
    if (!visible) return;
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

  const setExplosion = (value) => {
    const nextExplosion = THREE.MathUtils.clamp(Number(value) || 0, 0, 1);

    explosionParts.forEach((part) => {
      part.mesh.position
        .copy(part.basePosition)
        .addScaledVector(part.direction, explosionDistance * nextExplosion);
    });

    if (cameraReady && nextExplosion !== appliedCameraExplosion) {
      const previousScale = 1 + appliedCameraExplosion * PART_EXPLOSION_CAMERA_SCALE;
      const nextScale = 1 + nextExplosion * PART_EXPLOSION_CAMERA_SCALE;
      const cameraOffset = camera.position.clone().sub(controls.target);

      camera.position
        .copy(controls.target)
        .addScaledVector(cameraOffset, nextScale / previousScale);
      controls.update();
    }

    currentExplosion = nextExplosion;
    appliedCameraExplosion = nextExplosion;
  };

  const update = async (item) => {
    const requestId = ++currentRequest;
    const loading = container.querySelector(".mesh-viewer__loading");
    const faces = slot.querySelector("[data-mesh-faces]");
    const vertices = slot.querySelector("[data-mesh-vertices]");
    const parts = slot.querySelector("[data-mesh-parts]");

    container.classList.remove("is-ready", "is-error");
    loading.textContent = "Loading mesh";
    if (faces) faces.textContent = "—";
    if (vertices) vertices.textContent = "—";
    if (parts) parts.textContent = "—";

    try {
      const gltf = await loader.loadAsync(item.path);
      if (requestId !== currentRequest) {
        disposeObject(gltf.scene);
        return;
      }

      const object = gltf.scene;
      const stats = prepareGlbScene(object, palette, {
        width: Math.max(container.clientWidth, 1),
        height: Math.max(container.clientHeight, 1),
      });
      if (!stats.parts) throw new Error("GLB contains no renderable mesh primitives");

      const initialBounds = new THREE.Box3().setFromObject(object);
      const center = initialBounds.getCenter(new THREE.Vector3());

      object.position.sub(center);
      object.updateMatrixWorld(true);

      const centeredBounds = new THREE.Box3().setFromObject(object);
      const sphere = centeredBounds.getBoundingSphere(new THREE.Sphere());
      const radius = Math.max(sphere.radius, 0.01);
      const nextExplosionParts = createExplosionParts(stats.meshes, radius);

      if (currentObject) {
        scene.remove(currentObject);
        disposeObject(currentObject);
      }

      currentObject = object;
      wireMaterials = stats.wireMaterials;
      explosionParts = nextExplosionParts;
      explosionDistance = radius * PART_EXPLOSION_DISTANCE_SCALE;
      scene.add(object);

      camera.near = Math.max(radius / 100, 0.001);
      camera.far = radius * 50;
      camera.position.set(radius * 2.55, radius * 1.25, radius * 2.85);
      camera.updateProjectionMatrix();

      controls.target.set(0, 0, 0);
      controls.minDistance = radius * 1.25;
      controls.maxDistance = radius * 12;
      controls.update();
      cameraReady = true;
      appliedCameraExplosion = 0;
      setExplosion(currentExplosion);

      if (faces) faces.textContent = formatCount(stats.faces);
      if (vertices) vertices.textContent = formatCount(stats.vertices);
      if (parts) parts.textContent = formatCount(stats.parts);
      container.dataset.glbPath = item.path;
      container.setAttribute("aria-label", `Interactive viewer for ${item.label}`);
      container.classList.add("is-ready");
    } catch (error) {
      if (requestId !== currentRequest) return;
      console.error(`Unable to load ${item.path}`, error);
      loading.textContent = "Mesh could not be loaded";
      container.classList.add("is-error");
    }
  };

  reducedMotion.addEventListener("change", (event) => {
    controls.autoRotate = !event.matches;
  });

  return { update, setExplosion };
};

const createEditViewer = (container) => {
  const palette = readPalette();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  const controls = new OrbitControls(camera, renderer.domElement);
  const objLoader = new OBJLoader();
  const plyLoader = new PLYLoader();

  let currentObject = null;
  let currentRequest = 0;
  let wireMaterials = [];
  let visible = true;

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0xf8f3ef, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.04;
  renderer.domElement.setAttribute("aria-hidden", "true");
  container.prepend(renderer.domElement);

  controls.enableDamping = true;
  controls.dampingFactor = 0.055;
  controls.enablePan = false;
  controls.autoRotate = !reducedMotion.matches;
  controls.autoRotateSpeed = 0.64;

  scene.add(new THREE.HemisphereLight(0xfff7f0, 0x667581, 2.15));

  const keyLight = new THREE.DirectionalLight(0xfff4e9, 3.1);
  keyLight.position.set(4, 6, 5);
  scene.add(keyLight);

  const rimLight = new THREE.DirectionalLight(0xb8cad4, 1.85);
  rimLight.position.set(-5, 2, -4);
  scene.add(rimLight);

  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    wireMaterials.forEach((material) => material.resolution.set(width, height));
  };

  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();

  if ("IntersectionObserver" in window) {
    const visibilityObserver = new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting ?? true;
    }, { rootMargin: "180px 0px" });
    visibilityObserver.observe(container);
  }

  const animate = () => {
    requestAnimationFrame(animate);
    if (!visible) return;
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

  const update = async (item) => {
    const requestId = ++currentRequest;
    const loading = container.querySelector(".mesh-viewer__loading");

    container.classList.remove("is-ready", "is-error");
    loading.textContent = "Loading mesh";

    try {
      const response = await fetch(item.path);
      if (!response.ok) throw new Error(`Editing mesh request failed with ${response.status}`);

      let object;
      let nextWireMaterials = [];

      if (item.type === "vertex") {
        const source = await response.arrayBuffer();
        const geometry = plyLoader.parse(source);
        object = createColoredPointCloud(geometry);
      } else {
        const source = await response.text();
        object = objLoader.parse(source);
        nextWireMaterials = prepareColoredObj(object, palette, {
          width: Math.max(container.clientWidth, 1),
          height: Math.max(container.clientHeight, 1),
        });
      }

      if (requestId !== currentRequest) {
        disposeObject(object);
        return;
      }

      const initialBounds = new THREE.Box3().setFromObject(object);
      const center = initialBounds.getCenter(new THREE.Vector3());

      object.position.sub(center);
      object.updateMatrixWorld(true);

      const centeredBounds = new THREE.Box3().setFromObject(object);
      const sphere = centeredBounds.getBoundingSphere(new THREE.Sphere());
      const radius = Math.max(sphere.radius, 0.01);

      object.traverse((child) => {
        if (child.isPoints) child.material.size = radius * EDIT_POINT_SIZE_SCALE;
      });

      if (currentObject) {
        scene.remove(currentObject);
        disposeObject(currentObject);
      }

      currentObject = object;
      wireMaterials = nextWireMaterials;
      scene.add(object);

      camera.near = Math.max(radius / 100, 0.001);
      camera.far = radius * 30;
      camera.position.set(radius * 2.55, radius * 1.25, radius * 2.85);
      camera.updateProjectionMatrix();

      controls.target.set(0, 0, 0);
      controls.minDistance = radius * 1.25;
      controls.maxDistance = radius * 8;
      controls.update();

      container.dataset.editPath = item.path;
      container.dataset.editType = item.type;
      container.dataset.editName = item.name;
      container.setAttribute("aria-label", `Interactive viewer for ${item.name}`);
      container.classList.add("is-ready");
    } catch (error) {
      if (requestId !== currentRequest) return;
      console.error(`Unable to load ${item.path}`, error);
      loading.textContent = "Mesh could not be loaded";
      container.classList.add("is-error");
    }
  };

  reducedMotion.addEventListener("change", (event) => {
    controls.autoRotate = !event.matches;
  });

  return { update };
};

const initializeSingleGeneration = async () => {
  const group = document.querySelector("[data-single-viewer-group]");
  if (!group) return;

  const slots = [...group.querySelectorAll("[data-viewer-slot]")];
  const viewers = slots.map((slot) => {
    const container = slot.querySelector("[data-obj-viewer]");
    return createObjViewer(container, slot);
  });
  const previousButton = group.querySelector("[data-viewer-prev]");
  const nextButton = group.querySelector("[data-viewer-next]");
  const counter = group.querySelector("[data-viewer-counter]");
  const status = group.querySelector("[data-viewer-status]");
  let currentSet = 0;

  previousButton.disabled = true;
  nextButton.disabled = true;

  const meshItems = await loadSingleMeshItems(group);
  const singleGenerationSets = groupItems(meshItems, MESHES_PER_SET);

  if (!singleGenerationSets.length) {
    counter.textContent = "No OBJ results";
    status.textContent = "No single-generation OBJ files were found.";
    slots.forEach((slot) => {
      slot.hidden = true;
    });
    return;
  }

  group.dataset.setCount = String(singleGenerationSets.length);
  previousButton.disabled = singleGenerationSets.length < 2;
  nextButton.disabled = singleGenerationSets.length < 2;

  const renderSet = () => {
    const items = singleGenerationSets[currentSet];

    counter.textContent = `Set ${formatIndex(currentSet + 1)} / ${formatIndex(singleGenerationSets.length)}`;
    group.dataset.currentSet = String(currentSet + 1);

    viewers.forEach((viewer, index) => {
      const item = items[index];
      const slot = slots[index];
      if (!slot) return;

      slot.hidden = !item;
      if (!item) return;

      const meshIndex = currentSet * MESHES_PER_SET + index + 1;
      slot.querySelector(".viewer-badge").textContent = `OBJ ${formatIndex(meshIndex)}`;
      viewer.update(item);
    });

    status.textContent = `Generation result set ${currentSet + 1} of ${singleGenerationSets.length} displayed.`;
  };

  previousButton.addEventListener("click", () => {
    currentSet = (currentSet - 1 + singleGenerationSets.length) % singleGenerationSets.length;
    renderSet();
  });

  nextButton.addEventListener("click", () => {
    currentSet = (currentSet + 1) % singleGenerationSets.length;
    renderSet();
  });

  renderSet();
};

const initializeMeshComplexityControl = async () => {
  const group = document.querySelector("[data-mesh-complexity-viewer-group]");
  if (!group) return;

  const slots = [...group.querySelectorAll("[data-viewer-slot]")];
  const viewers = slots.map((slot) => {
    const container = slot.querySelector("[data-mesh-complexity-viewer]");
    return createObjViewer(container, slot);
  });
  const previousButton = group.querySelector("[data-viewer-prev]");
  const nextButton = group.querySelector("[data-viewer-next]");
  const counter = group.querySelector("[data-viewer-counter]");
  const status = group.querySelector("[data-viewer-status]");
  let currentPage = 0;

  previousButton.disabled = true;
  nextButton.disabled = true;

  const controlSets = await loadControlSets(group);

  if (!controlSets.length) {
    counter.textContent = "No OBJ results";
    status.textContent = "No mesh-complexity OBJ files were found.";
    slots.forEach((slot) => {
      slot.hidden = true;
    });
    return;
  }

  group.dataset.setCount = String(controlSets.length);
  previousButton.disabled = controlSets.length < 2;
  nextButton.disabled = controlSets.length < 2;

  const renderPage = () => {
    const controlSet = controlSets[currentPage];

    counter.textContent = `Page ${formatIndex(currentPage + 1)} / ${formatIndex(controlSets.length)}`;
    group.dataset.currentSet = String(currentPage + 1);

    viewers.forEach((viewer, index) => {
      const item = controlSet.items[index];
      const slot = slots[index];
      if (!slot) return;

      slot.hidden = !item;
      if (!item) return;

      const budget = formatCount(item.budget);
      slot.querySelector(".viewer-badge").textContent = `${budget} vertices`;
      viewer.update(item);
    });

    const pageName = controlSet.name ? ` (${controlSet.name})` : "";
    status.textContent = `Mesh-complexity page ${currentPage + 1} of ${controlSets.length}${pageName} displayed.`;
  };

  previousButton.addEventListener("click", () => {
    currentPage = (currentPage - 1 + controlSets.length) % controlSets.length;
    renderPage();
  });

  nextButton.addEventListener("click", () => {
    currentPage = (currentPage + 1) % controlSets.length;
    renderPage();
  });

  renderPage();
};

const initializePartwiseGeneration = async () => {
  const group = document.querySelector("[data-multi-viewer-group]");
  if (!group) return;

  const slot = group.querySelector("[data-viewer-slot]");
  const container = slot?.querySelector("[data-multi-viewer]");
  if (!slot || !container) return;

  const viewer = createGlbViewer(container, slot);
  const previousButton = group.querySelector("[data-viewer-prev]");
  const nextButton = group.querySelector("[data-viewer-next]");
  const counter = group.querySelector("[data-viewer-counter]");
  const status = group.querySelector("[data-viewer-status]");
  const badge = slot.querySelector(".viewer-badge");
  const explosionInput = slot.querySelector("[data-part-explode]");
  const explosionValue = slot.querySelector("[data-part-explode-value]");
  let currentResult = 0;

  previousButton.disabled = true;
  nextButton.disabled = true;
  explosionInput.disabled = true;

  const setExplosionValue = (value) => {
    const percentage = THREE.MathUtils.clamp(Math.round(Number(value) || 0), 0, 100);

    explosionInput.value = String(percentage);
    explosionInput.style.setProperty("--explode-progress", `${percentage}%`);
    explosionInput.setAttribute("aria-valuetext", `${percentage}% exploded`);
    explosionValue.textContent = `${percentage}%`;
    viewer.setExplosion(percentage / 100);
  };

  explosionInput.addEventListener("input", () => {
    setExplosionValue(explosionInput.value);
  });

  const meshItems = await loadMultiMeshItems(group);

  if (!meshItems.length) {
    counter.textContent = "No GLB results";
    status.textContent = "No part-wise GLB files were found.";
    slot.hidden = true;
    return;
  }

  group.dataset.setCount = String(meshItems.length);
  previousButton.disabled = meshItems.length < 2;
  nextButton.disabled = meshItems.length < 2;
  explosionInput.disabled = false;

  const renderResult = () => {
    const item = meshItems[currentResult];
    const resultNumber = formatIndex(currentResult + 1);

    counter.textContent = `Result ${resultNumber} / ${formatIndex(meshItems.length)}`;
    group.dataset.currentSet = String(currentResult + 1);
    badge.textContent = `Result ${resultNumber}`;
    setExplosionValue(0);
    viewer.update(item);
    status.textContent = `Part-wise generation result ${currentResult + 1} of ${meshItems.length} displayed.`;
  };

  previousButton.addEventListener("click", () => {
    currentResult = (currentResult - 1 + meshItems.length) % meshItems.length;
    renderResult();
  });

  nextButton.addEventListener("click", () => {
    currentResult = (currentResult + 1) % meshItems.length;
    renderResult();
  });

  renderResult();
};

const initializeTopologyEditing = async () => {
  const group = document.querySelector("[data-edit-viewer-group]");
  if (!group) return;
  const setName = document.querySelector("[data-edit-set-name]");

  const slots = [...group.querySelectorAll("[data-viewer-slot]")];
  const viewers = slots.map((slot) => {
    const container = slot.querySelector("[data-edit-viewer]");
    return createEditViewer(container);
  });
  const previousButton = group.querySelector("[data-viewer-prev]");
  const nextButton = group.querySelector("[data-viewer-next]");
  const counter = group.querySelector("[data-viewer-counter]");
  const status = group.querySelector("[data-viewer-status]");
  let currentSet = 0;

  previousButton.disabled = true;
  nextButton.disabled = true;

  const editSets = await loadEditSets(group);

  if (!editSets.length) {
    counter.textContent = "No editing results";
    status.textContent = "No topology-editing meshes were found.";
    slots.forEach((slot) => {
      slot.hidden = true;
    });
    return;
  }

  group.dataset.setCount = String(editSets.length);
  previousButton.disabled = editSets.length < 2;
  nextButton.disabled = editSets.length < 2;

  const renderSet = () => {
    const editSet = editSets[currentSet];

    counter.textContent = `Result ${formatIndex(currentSet + 1)} / ${formatIndex(editSets.length)}`;
    group.dataset.currentSet = String(currentSet + 1);
    if (setName) {
      setName.textContent = editSet.name || `Result ${formatIndex(currentSet + 1)}`;
    }

    viewers.forEach((viewer, index) => {
      const item = editSet.items[index];
      const slot = slots[index];
      if (!slot) return;

      slot.hidden = !item;
      if (!item) return;

      slot.querySelector(".viewer-badge").textContent = item.type === "vertex"
        ? "Vertices"
        : "Mesh";
      slot.querySelector("[data-edit-label]").textContent = item.name;
      viewer.update(item);
    });

    const statusName = editSet.name ? ` (${editSet.name})` : "";
    status.textContent = `Topology-editing result ${currentSet + 1} of ${editSets.length}${statusName} displayed.`;
  };

  previousButton.addEventListener("click", () => {
    currentSet = (currentSet - 1 + editSets.length) % editSets.length;
    renderSet();
  });

  nextButton.addEventListener("click", () => {
    currentSet = (currentSet + 1) % editSets.length;
    renderSet();
  });

  renderSet();
};

initializeSingleGeneration();
initializeMeshComplexityControl();
initializePartwiseGeneration();
initializeTopologyEditing();
