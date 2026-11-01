import { afterEach, beforeEach, it } from "node:test";
import assert from "node:assert/strict";
import { registerHooks } from "node:module";

let startupSdkImports = 0;
const hook = registerHooks({ resolve(specifier, context, next) {
	if (specifier.includes("@anthropic-ai/claude-agent-sdk")) startupSdkImports++;
	return next(specifier, context);
} });
const mod = await import("../src/index.js");
hook.deregister();
const { __test } = mod;
let provider;
const handlers = {};
mod.default({ on: (name, fn) => { (handlers[name] ??= []).push(fn); }, registerTool: () => {}, registerProvider: (_id, value) => { provider = value; } });
const model = provider.models[0];
let serial = 0;
let calls;
const tick = () => new Promise(setImmediate);
const event = (value) => ({ type: "stream_event", event: value });

function query(params) {
	calls.push(params);
	const output = (async function* () {
		const id = params.options.sessionId ?? params.options.resume ?? "summary";
		yield { type: "system", subtype: "init", session_id: id };
		if (typeof params.prompt !== "string") {
			const first = await params.prompt[Symbol.asyncIterator]().next();
			yield { ...first.value, type: "user", isReplay: true, session_id: id };
		}
		yield event({ type: "message_start", message: { id: "message", usage: {} } });
		yield event({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
		yield event({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "stub answer" } });
		yield event({ type: "content_block_stop", index: 0 });
		yield event({ type: "message_delta", delta: { stop_reason: "end_turn" }, usage: {} });
		yield event({ type: "message_stop" });
		yield { type: "result", subtype: "success", result: "stub answer", is_error: false };
	})();
	output.interrupt = async () => {};
	output.close = () => {};
	return output;
}

function start(kind, signal, sessionId = `lazy-${++serial}`) {
	const task = kind === "ask"
		? __test.promptAndWait("question", "none", new Map(), signal, { model: model.id, appendSkills: false, piSessionId: sessionId })
		: provider.streamSimple(model, { messages: [{ role: "user", content: "question", timestamp: 1 }], tools: [] },
			{ sessionId, signal, ...(kind === "summary" ? { cacheRetention: "none" } : {}) }).result();
	// Observe rejections immediately; calls may finish before a sibling loads.
	return { kind, result: task.then((output) => ({ output }), (error) => ({ error })) };
}
function assertOk({ output, error }) {
	assert.equal(error, undefined);
	assert.equal(output.stopReason, "stop", output.errorMessage);
	const text = output.responseText ?? output.content.map((block) => block.text ?? "").join("");
	assert.equal(text, "stub answer");
}
function assertAborted(kind, { output, error }) {
	if (kind === "ask") assert.match(error?.message ?? "", /Aborted/);
	else assert.equal(output?.stopReason, "aborted", output?.errorMessage);
}
function holdImport() {
	const started = Promise.withResolvers();
	const module = Promise.withResolvers();
	let imports = 0;
	__test.setSdkLoader(() => { imports++; started.resolve(); return module.promise; });
	return { started: started.promise, release: () => module.resolve({ query }), imports: () => imports };
}

beforeEach(() => { calls = []; __test.setQuery(null); __test.resetSharedSession(); __test.setSdkLoader(null); });
afterEach(() => { __test.resetSharedSession(); __test.setQuery(null); __test.setSdkLoader(null); });
it("registers the bridge without importing the Claude SDK", () => assert.equal(startupSdkImports, 0));

for (const kind of ["provider", "ask", "summary"]) {
	it(`${kind}: abort while importing ends before import completion and never starts a late query`, { timeout: 5000 }, async () => {
		const loading = holdImport();
		const controller = new AbortController();
		const call = start(kind, controller.signal);
		await loading.started;
		controller.abort();
		assertAborted(kind, await call.result);
		assert.equal(calls.length, 0);
		loading.release();
		await tick();
		assert.equal(calls.length, 0);
		assert.equal(loading.imports(), 1);
		assertOk(await start(kind).result);
		assert.equal(loading.imports(), 1);
	});
	it(`${kind}: an already-aborted call does not import or start the SDK`, { timeout: 5000 }, async () => {
		const loading = holdImport();
		const controller = new AbortController();
		controller.abort();
		assertAborted(kind, await start(kind, controller.signal).result);
		assert.equal(loading.imports(), 0);
		assert.equal(calls.length, 0);
	});
	it(`${kind}: reports an import failure and allows the next call to retry`, { timeout: 5000 }, async () => {
		let imports = 0;
		__test.setSdkLoader(async () => { if (++imports === 1) throw new Error("SDK import failed"); return { query }; });
		const { output, error } = await start(kind).result;
		assert.match(error?.message ?? output?.errorMessage ?? "", /SDK import failed/);
		assert.equal(calls.length, 0);
		assertOk(await start(kind).result);
		assert.equal(imports, 2);
	});
}

it("all three first callers share an import; aborting one does not cancel its siblings", { timeout: 5000 }, async () => {
	const loading = holdImport();
	const controller = new AbortController();
	const cancelled = start("provider", controller.signal);
	const siblings = [start("ask"), start("summary")];
	await loading.started;
	await tick();
	assert.equal(loading.imports(), 1);
	controller.abort();
	assertAborted("provider", await cancelled.result);
	loading.release();
	for (const sibling of siblings) assertOk(await sibling.result);
	assert.equal(calls.length, 2);
	assert.equal(loading.imports(), 1);
});

it("a query test double does not trigger an SDK import", async () => {
	const loading = holdImport();
	__test.setQuery(query);
	for (const kind of ["provider", "ask", "summary"]) assertOk(await start(kind).result);
	assert.equal(loading.imports(), 0);
});

it("a fresh request replacing the same session during cold import does not launch the discarded query", { timeout: 5000 }, async () => {
	const loading = holdImport();
	const first = start("provider", undefined, "replaced-during-import");
	await loading.started;
	const replacement = start("provider", undefined, "replaced-during-import");
	await tick();
	loading.release();
	await first.result;
	assertOk(await replacement.result);
	await tick();
	assert.equal(calls.length, 1, "only the replacement may start Claude");
});

it("closing a session during cold import prevents its launch without cancelling another session", { timeout: 5000 }, async () => {
	const loading = holdImport();
	const closed = start("provider", undefined, "closed-during-import");
	await loading.started;
	const sibling = start("provider", undefined, "unrelated-session");
	for (const fn of handlers.session_shutdown) fn({}, { sessionManager: { getSessionId: () => "closed-during-import" } });
	const { output } = await closed.result;
	assert.equal(output.stopReason, "error");
	assert.match(output.errorMessage, /Conversation replaced/);
	loading.release();
	assertOk(await sibling.result);
	await tick();
	assert.equal(calls.length, 1, "only the other session may launch");
});
