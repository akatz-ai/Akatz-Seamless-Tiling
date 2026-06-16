const { app } = window.comfyAPI.app;

const SET_NODE_TYPE = "AkatzSetNode";
const GET_NODE_TYPE = "AkatzGetNode";
const AUX_ID = "akatz-ai/Akatz-Seamless-Tiling";
const LGraphNode = LiteGraph.LGraphNode;

function getLink(graph, linkId) {
	if (linkId == null || !graph) return null;
	if (graph.links?.[linkId]) return graph.links[linkId];
	if (graph._links?.[linkId]) return graph._links[linkId];
	if (Array.isArray(graph.links)) return graph.links.find((link) => link?.id === linkId) ?? null;
	return null;
}

function nodeName(node) {
	return node?.widgets?.[0]?.value ?? "";
}

function nodesOfType(graph, type) {
	return (graph?._nodes || []).filter((node) => node.type === type);
}

function typeFromLink(graph, linkId) {
	const link = getLink(graph, linkId);
	if (!link) return "*";
	if (link.type && link.type !== "*") return link.type;
	const sourceNode = graph?.getNodeById?.(link.origin_id);
	return sourceNode?.outputs?.[link.origin_slot]?.type || "*";
}

function setterInputType(setter) {
	const input = setter?.inputs?.[0];
	if (!input) return "*";
	if (input.link != null) return typeFromLink(setter.graph, input.link);
	return input.type || "*";
}

function findSetterByName(graph, name) {
	if (!graph || !name) return null;
	return nodesOfType(graph, SET_NODE_TYPE).find((node) => nodeName(node) === name) ?? null;
}

function visibleSetterNames(graph) {
	return nodesOfType(graph, SET_NODE_TYPE)
		.map((node) => nodeName(node))
		.filter(Boolean)
		.sort((a, b) => a.localeCompare(b));
}

function applyType(node, type, direction) {
	const cleanType = type || "*";
	const slot = direction === "input" ? node.inputs?.[0] : node.outputs?.[0];
	if (!slot) return;
	slot.name = cleanType;
	slot.type = cleanType;
}

function ensureVirtualNode(node, nodeType) {
	node.properties = node.properties || {};
	node.properties["Node name for S&R"] = nodeType;
	node.properties.aux_id = AUX_ID;
	node.isVirtualNode = true;
	node.serialize_widgets = true;
	node.drawConnection = false;
}

function updateGetter(getter) {
	const name = nodeName(getter);
	const setter = findSetterByName(getter.graph, name);
	const type = setter ? setterInputType(setter) : "*";
	applyType(getter, type, "output");
	getter.title = name ? `Get_${name}` : "Get";
}

function updateGettersForName(graph, name) {
	if (!graph || !name) return;
	for (const getter of nodesOfType(graph, GET_NODE_TYPE)) {
		if (nodeName(getter) === name) updateGetter(getter);
	}
}

function markCanvasDirty() {
	app.canvas?.setDirty(true, true);
}

app.registerExtension({
	name: "Akatz.SeamlessTiling.SetGet",
	registerCustomNodes() {
		class AkatzSetNode extends LGraphNode {
			static title = "Set";
			static category = "Akatz";

			constructor(title) {
				super(title);
				ensureVirtualNode(this, SET_NODE_TYPE);
				this.addWidget("text", "name", "", () => this.onRename());
				this.addInput("*", "*");
				this.addOutput("*", "*");
			}

			onConfigure() {
				ensureVirtualNode(this, SET_NODE_TYPE);
				setTimeout(() => this.updateType(), 0);
			}

			onAdded() {
				this.updateType();
			}

			onConnectionsChange() {
				if (!app.configuringGraph) this.updateType();
			}

			onRename() {
				this.updateType();
			}

			updateType() {
				const name = nodeName(this);
				const previousName = this.properties.previousName;
				this.properties.previousName = name;
				this.title = name ? `Set_${name}` : "Set";
				const type = setterInputType(this);
				applyType(this, type, "input");
				applyType(this, type, "output");
				updateGettersForName(this.graph, name);
				if (previousName && previousName !== name) updateGettersForName(this.graph, previousName);
				markCanvasDirty();
			}
		}

		class AkatzGetNode extends LGraphNode {
			static title = "Get";
			static category = "Akatz";

			constructor(title) {
				super(title);
				ensureVirtualNode(this, GET_NODE_TYPE);
				const options = {};
				Object.defineProperty(options, "values", {
					get: () => visibleSetterNames(this.graph),
					enumerable: true,
				});
				this.addWidget("combo", "name", "", () => this.onRename(), options);
				this.addOutput("*", "*");
			}

			onConfigure() {
				ensureVirtualNode(this, GET_NODE_TYPE);
				setTimeout(() => this.onRename(), 0);
			}

			onAdded() {
				this.onRename();
			}

			onRename() {
				updateGetter(this);
				markCanvasDirty();
			}
		}

		LiteGraph.registerNodeType(SET_NODE_TYPE, AkatzSetNode);
		LiteGraph.registerNodeType(GET_NODE_TYPE, AkatzGetNode);
		AkatzSetNode.category = "Akatz";
		AkatzGetNode.category = "Akatz";
	},
});

