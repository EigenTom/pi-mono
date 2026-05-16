/*
	Pull-style tools for organized corpus workspaces.
	The agent chooses semantic queries, and pull materializes retrieved
	documents into the visible workspace for local search/read.
*/

import { access, link, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, relative, resolve } from "node:path";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import fetch, { type RequestInit, type Response } from "node-fetch";
import type { ToolDefinition, ToolRenderResultOptions } from "../extensions/types.js";
import { getTextOutput } from "./render-utils.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

const pullSchema = Type.Object({
	queries: Type.Array(Type.String({ minLength: 1 }), {
		minItems: 1,
		maxItems: 8,
		description: "Focused semantic queries used to pull relevant documents into the workspace.",
	}),
	topK: Type.Integer({
		minimum: 100,
		maximum: 500,
		description: "Number of documents to retrieve for each query. Choose 100-500 per query.",
	}),
});

const rankAwarePullSchema = Type.Object({
	query: Type.String({
		minLength: 1,
		description: "One concise lexical query. Call pull again for a different clue.",
	}),
	topK: Type.Integer({
		minimum: 100,
		maximum: 500,
		description: "Number of documents to retrieve for this query. Choose 100-500.",
	}),
});

export type PullToolInput = Static<typeof pullSchema>;
type RankAwarePullToolInput = Static<typeof rankAwarePullSchema>;
type PullExecuteInput = PullToolInput | RankAwarePullToolInput;

type RetrieverResult = {
	docid?: string;
	doc_path: string;
	score: number;
};

type PullMaterializationMode = "original" | "ranked" | "ranked_flat";

function readPositiveIntEnv(name: string): number | undefined {
	const raw = process.env[name];
	if (!raw) return undefined;
	const value = Number.parseInt(raw, 10);
	return Number.isFinite(value) && value > 0 ? value : undefined;
}

export interface PullToolDetails {
	toolKind: "pull";
	queries: string[];
	topK: number;
	groups?: Array<{ topic: string; queries: string[]; topK: number; dir: string }>;
	viewMode: "hardlink";
	layout: "query" | "pull";
	materializationMode?: PullMaterializationMode;
	viewDir: string;
	pullIndex: number;
	pullDir: string;
	workspaceDir: string;
	managedPathsPath: string;
	sourceDocumentCount: number;
	materializedDocumentCount: number;
	missingDocumentCount: number;
	perQueryHitCounts: Record<string, number>;
	queryDirs: Record<string, string>;
}

export interface PullOperations {
	fetch: (url: string, options: RequestInit) => Promise<Response>;
	writeFile: (path: string, content: string) => Promise<void>;
	readFile: (path: string) => Promise<string>;
	readdir: (path: string) => Promise<string[]>;
	mkdir: (path: string) => Promise<void>;
	link: (existingPath: string, newPath: string) => Promise<void>;
}

const defaultPullOperations: PullOperations = {
	fetch: (url, options) => fetch(url, options),
	writeFile: (path, content) => writeFile(path, content, "utf-8"),
	readFile: (path) => readFile(path, "utf-8"),
	readdir: (path) => readdir(path),
	mkdir: (path) => mkdir(path, { recursive: true }).then(() => {}),
	link: (existingPath, newPath) => link(existingPath, newPath),
};

export interface PullToolOptions {
	baseUrl?: string;
	viewDir?: string;
	sourceRoot?: string;
	operations?: Partial<PullOperations>;
}

const MANAGED_PATHS_FILE = "managed_paths.json";
const SUBMIT_NOW_MARKER = ".dci_budget/submit_now.json";

function safeRelativePath(docPath: string): string | undefined {
	const normalized = normalize(docPath.replace(/\\/g, "/"));
	if (!normalized || normalized === "." || isAbsolute(normalized)) return undefined;
	if (normalized === ".." || normalized.startsWith("..")) return undefined;
	return normalized;
}

function isInside(parent: string, child: string): boolean {
	const rel = relative(parent, child);
	return rel === "" || (!!rel && !rel.startsWith("..") && !isAbsolute(rel));
}

function slugify(value: string): string {
	const slug = value
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "")
		.slice(0, 48);
	return slug || "query";
}

function safeFilename(value: string): string {
	const normalized = value.replace(/\\/g, "/");
	const lastSlash = normalized.lastIndexOf("/");
	const basename = lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized;
	const lastDot = basename.lastIndexOf(".");
	const rawStem = lastDot > 0 ? basename.slice(0, lastDot) : basename;
	const rawExt = lastDot > 0 ? basename.slice(lastDot + 1) : "";
	const stem =
		rawStem
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, "_")
			.replace(/^_+|_+$/g, "")
			.slice(0, 96) || "document";
	const ext = rawExt
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "")
		.slice(0, 16);
	return ext ? `${stem}.${ext}` : stem;
}

function rankPrefixedRelativePath(safePath: string, rank: number): string {
	const normalized = safePath.replace(/\\/g, "/");
	const lastSlash = normalized.lastIndexOf("/");
	const parent = lastSlash >= 0 ? normalized.slice(0, lastSlash) : "";
	const basename = lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized;
	const prefixed = `${String(rank).padStart(4, "0")}__${safeFilename(basename)}`;
	return parent ? `${parent}/${prefixed}` : prefixed;
}

function rankPrefixedFlatPath(safePath: string, rank: number): string {
	const normalized = safePath.replace(/\\/g, "/");
	const lastSlash = normalized.lastIndexOf("/");
	const basename = lastSlash >= 0 ? normalized.slice(lastSlash + 1) : normalized;
	return `${String(rank).padStart(4, "0")}__${safeFilename(basename)}`;
}

function parsePullMaterializationMode(value: string | undefined): PullMaterializationMode {
	if (value === "ranked_flat") return "ranked_flat";
	if (value === "ranked") return "ranked";
	return "original";
}

function isEnabledEnv(name: string): boolean {
	const raw = process.env[name]?.toLowerCase();
	return raw === "1" || raw === "true" || raw === "yes";
}

function reflowSingleLineText(text: string, width: number): string {
	const normalized = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
	const chunks = normalized.replace(/([.!?。！？；;])\s+/g, "$1\n").split("\n");
	const lines: string[] = [];
	for (const chunk of chunks) {
		const trimmed = chunk.trim();
		if (!trimmed) continue;
		for (let offset = 0; offset < trimmed.length; offset += width) {
			lines.push(trimmed.slice(offset, offset + width));
		}
	}
	return `${lines.join("\n")}\n`;
}

async function materializeFile(args: { sourcePath: string; targetPath: string; ops: PullOperations }): Promise<void> {
	const { sourcePath, targetPath, ops } = args;
	if (!isEnabledEnv("DCI_REFLOW_SINGLE_LINE_TEXT")) {
		await ops.link(sourcePath, targetPath);
		return;
	}

	const width = readPositiveIntEnv("DCI_REFLOW_SINGLE_LINE_WIDTH") ?? 1_200;
	const text = await ops.readFile(sourcePath);

	const firstNewline = text.indexOf("\n");
	const secondNewline = firstNewline >= 0 ? text.indexOf("\n", firstNewline + 1) : -1;
	if (firstNewline >= 0 && secondNewline >= 0) {
		await ops.link(sourcePath, targetPath);
		return;
	}

	const minBytes = readPositiveIntEnv("DCI_REFLOW_SINGLE_LINE_MIN_BYTES");
	if (minBytes !== undefined && Buffer.byteLength(text, "utf8") < minBytes) {
		await ops.link(sourcePath, targetPath);
		return;
	}

	await ops.writeFile(targetPath, reflowSingleLineText(text, width));
}

async function submitNowMarkerExists(viewDir: string): Promise<boolean> {
	try {
		await access(join(viewDir, SUBMIT_NOW_MARKER));
		return true;
	} catch {
		return false;
	}
}

async function nextPullIndex(viewDir: string, ops: PullOperations): Promise<number> {
	try {
		const names = await ops.readdir(viewDir);
		const indices = names
			.map((name) => /^pull_(\d+)$/.exec(name)?.[1])
			.filter((value): value is string => !!value)
			.map((value) => Number.parseInt(value, 10))
			.filter((value) => Number.isFinite(value));
		return indices.length > 0 ? Math.max(...indices) + 1 : 1;
	} catch {
		return 1;
	}
}

async function retrieveOne(
	baseUrl: string,
	query: string,
	topK: number,
	ops: PullOperations,
	signal?: AbortSignal,
): Promise<RetrieverResult[]> {
	const response = await ops.fetch(baseUrl, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ query, top_k: topK }),
		signal,
	});
	if (!response.ok) {
		const text = await response.text();
		throw new Error(`Pull API error: ${response.status} ${response.statusText} - ${text}`);
	}
	const data = (await response.json()) as { results?: RetrieverResult[] };
	return Array.isArray(data.results) ? data.results : [];
}

async function materializeQueryDocs(args: {
	query: string;
	queryIndex: number;
	hits: RetrieverResult[];
	pullDir: string;
	sourceRoot: string;
	ops: PullOperations;
	layout: "query" | "pull";
	materializationMode: PullMaterializationMode;
	createdSet?: Set<string>;
}): Promise<{ queryDir: string; created: string[]; missing: string[] }> {
	const { query, queryIndex, hits, pullDir, sourceRoot, ops, layout, materializationMode } = args;
	const queryDirName = `q${String(queryIndex + 1).padStart(2, "0")}_${slugify(query)}`;
	const queryDir = layout === "query" ? join(pullDir, queryDirName) : pullDir;
	const created: string[] = [];
	const missing: string[] = [];
	const createdSet = args.createdSet ?? new Set<string>();
	await ops.mkdir(queryDir);

	for (const [hitIndex, hit] of hits.entries()) {
		const safePath = safeRelativePath(hit.doc_path);
		if (!safePath) {
			missing.push(hit.doc_path);
			continue;
		}
		if (createdSet.has(safePath)) continue;

		const sourcePath = resolve(sourceRoot, safePath);
		if (!isInside(sourceRoot, sourcePath)) {
			missing.push(hit.doc_path);
			continue;
		}

		const workspacePath =
			materializationMode === "ranked_flat"
				? rankPrefixedFlatPath(safePath, hitIndex + 1)
				: materializationMode === "ranked"
					? rankPrefixedRelativePath(safePath, hitIndex + 1)
					: safePath;
		const targetPath = resolve(queryDir, workspacePath);
		if (!isInside(queryDir, targetPath)) {
			missing.push(hit.doc_path);
			continue;
		}

		try {
			await ops.mkdir(dirname(targetPath));
			await materializeFile({ sourcePath, targetPath, ops });
			created.push(safePath);
			createdSet.add(safePath);
		} catch {
			missing.push(hit.doc_path);
		}
	}

	return { queryDir, created, missing };
}

function formatPullCall(
	args: { query?: string; queries?: string[]; topK?: number } | undefined,
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
	toolName = "pull",
): string {
	const topK = args?.topK ?? "?";
	if (typeof args?.query === "string") {
		return `${theme.fg("toolTitle", theme.bold(toolName))} ${theme.fg("toolOutput", `1 query x ${topK}`)}`;
	}
	const queryCount = args?.queries?.length ?? 0;
	return `${theme.fg("toolTitle", theme.bold(toolName))} ${theme.fg("toolOutput", `${queryCount} queries x ${topK}`)}`;
}

function formatPullResult(
	result: { content: Array<{ type: string; text?: string; data?: string; mimeType?: string }> },
	_options: ToolRenderResultOptions,
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
): string {
	return `\n${theme.fg("toolOutput", getTextOutput(result, false))}`;
}

export function createPullToolDefinition(cwd: string, options?: PullToolOptions): ToolDefinition<any, PullToolDetails> {
	const baseUrl = options?.baseUrl ?? process.env.DCI_PULL_BASE_URL ?? "http://localhost:8000/retrieve";
	const viewDir = resolve(cwd, options?.viewDir ?? process.env.DCI_PULL_VIEW_DIR ?? ".");
	const metaBaseDir = process.env.DCI_PULL_META_DIR
		? resolve(process.env.DCI_PULL_META_DIR)
		: join(viewDir, ".dci_pull_meta");
	const sourceRoot = options?.sourceRoot ?? process.env.DCI_PULL_SOURCE_ROOT;
	const layout = process.env.DCI_PULL_LAYOUT === "pull" ? "pull" : "query";
	const promptMode = process.env.DCI_PULL_PROMPT_MODE === "rank_aware" ? "rank_aware" : "default";
	const materializationMode = parsePullMaterializationMode(process.env.DCI_PULL_MATERIALIZATION_MODE);
	const rankAwareMode = promptMode === "rank_aware" || materializationMode !== "original";
	const harnessTopK = readPositiveIntEnv("DCI_PULL_TOP_K");
	const parameters = rankAwareMode ? rankAwarePullSchema : pullSchema;
	const ops = { ...defaultPullOperations, ...options?.operations };
	const toolName = "pull";
	const label = "Pull Corpus Documents";
	const description = rankAwareMode
		? "Pull semantically relevant documents from the hidden corpus into the visible workspace. Rank-aware mode accepts exactly one query and topK 100-500 per call."
		: "Pull semantically relevant documents from the hidden corpus into the visible workspace, organized by pull call.";
	const folderDescription =
		layout === "pull"
			? rankAwareMode
				? "pull(query, topK) retrieves semantically relevant documents from the hidden corpus into the visible workspace. It accepts one query string per call and topK must be 100-500. Each call creates ./pull_N/ and stores rank-prefixed files directly inside it; lower rank numbers are more similar to the query."
				: "pull(queries, topK) retrieves semantically relevant documents from the hidden corpus into the visible workspace. Each call creates ./pull_N/ and stores retrieved files directly inside it."
			: rankAwareMode
				? "pull(query, topK) retrieves semantically relevant documents from the hidden corpus into the visible workspace. It accepts one query string per call and topK must be 100-500. Each call creates ./pull_N/ and stores rank-prefixed files under that workspace; lower rank numbers are more similar to the query."
				: "pull(queries, topK) retrieves semantically relevant documents from the hidden corpus into the visible workspace. Each call creates ./pull_N/ with one subfolder per query.";

	const promptSnippet = folderDescription;
	const promptGuidelines = [
		"The visible workspace starts empty; pull adds documents from the hidden corpus.",
		rankAwareMode
			? "The query parameter is a single string. Choose topK between 100 and 500 for each call."
			: "topK is clamped to 100-500 documents per query.",
		...(promptMode === "rank_aware"
			? [
					"Rank-aware mode accepts one query string and topK; call pull again for a different clue.",
					"Rank prefixes indicate retrieval order within that query; lower numbers are more similar.",
				]
			: []),
		...(layout === "pull"
			? ["Each pull call creates ./pull_N/ and stores retrieved files directly inside it."]
			: ["Each pull call creates ./pull_N/ with one subfolder per query."]),
		"pull is not evidence. Final answers must come from document text actually searched or read in the workspace.",
	];

	return {
		name: toolName,
		label,
		description,
		promptSnippet,
		promptGuidelines,
		parameters,

		async execute(_toolCallId, params: PullExecuteInput, signal?: AbortSignal) {
			if (await submitNowMarkerExists(viewDir)) {
				throw new Error(
					"Budget limit reached. pull is disabled. Do not retrieve more documents. Answer now using the existing workspace.",
				);
			}
			if (!sourceRoot) {
				throw new Error("DCI_PULL_SOURCE_ROOT is required for pull hardlink mode");
			}
			if (isInside(resolve(sourceRoot), viewDir) || isInside(viewDir, resolve(sourceRoot))) {
				throw new Error("Pull viewDir must be separate from DCI_PULL_SOURCE_ROOT");
			}

			const rawQueries =
				rankAwareMode && "query" in params
					? [params.query]
					: "queries" in params && Array.isArray(params.queries)
						? params.queries
						: [];
			const queries = Array.from(new Set(rawQueries.map((query) => query.trim()).filter(Boolean)));
			if (queries.length === 0) {
				throw new Error(
					rankAwareMode ? "A non-empty query string is required" : "At least one non-empty query is required",
				);
			}
			if (rankAwareMode) {
				if (queries.length !== 1) {
					throw new Error(
						"Rank-aware pull requires exactly one query string. Call pull again for a different clue.",
					);
				}
			}
			const requestedTopK = "topK" in params ? params.topK : 100;
			const topK = rankAwareMode
				? (harnessTopK ?? Math.max(100, Math.min(500, requestedTopK)))
				: Math.max(100, Math.min(500, requestedTopK));
			const pullIndex = await nextPullIndex(viewDir, ops);
			const pullDir = join(viewDir, `pull_${pullIndex}`);
			const metaDir = join(metaBaseDir, `pull_${pullIndex}`);
			await ops.mkdir(metaDir);

			const perQueryHits: Record<string, RetrieverResult[]> = {};
			const queryDirs: Record<string, string> = {};
			const managedSourcePaths = new Set<string>();
			const createdSet = new Set<string>();
			let materializedDocumentCount = 0;
			let missingDocumentCount = 0;

			for (const [queryIndex, query] of queries.entries()) {
				if (signal?.aborted) throw new Error("Operation aborted");
				const hits = await retrieveOne(baseUrl, query, topK, ops, signal);
				perQueryHits[query] = hits;
				const materialized = await materializeQueryDocs({
					query,
					queryIndex,
					hits,
					pullDir,
					sourceRoot: resolve(sourceRoot),
					ops,
					layout,
					materializationMode,
					createdSet,
				});
				queryDirs[query] = relative(viewDir, materialized.queryDir).replace(/\\/g, "/");
				for (const created of materialized.created) {
					managedSourcePaths.add(created);
				}
				materializedDocumentCount += materialized.created.length;
				missingDocumentCount += materialized.missing.length;
			}

			const managedPathsPath = join(metaDir, MANAGED_PATHS_FILE);
			await ops.writeFile(managedPathsPath, JSON.stringify(Array.from(managedSourcePaths).sort(), null, 2));

			const perQueryHitCounts = Object.fromEntries(
				Object.entries(perQueryHits).map(([query, hits]) => [query, hits.length]),
			);
			const details: PullToolDetails = {
				toolKind: "pull",
				queries,
				topK,
				viewMode: "hardlink",
				layout,
				materializationMode,
				viewDir,
				pullIndex,
				pullDir,
				workspaceDir: relative(viewDir, pullDir).replace(/\\/g, "/"),
				managedPathsPath,
				sourceDocumentCount: managedSourcePaths.size,
				materializedDocumentCount,
				missingDocumentCount,
				perQueryHitCounts,
				queryDirs,
			};

			return {
				content: [
					{
						type: "text",
						text:
							layout === "pull"
								? materializationMode !== "original"
									? `Workspace expanded under ./pull_${pullIndex}. Documents are directly inside that folder and rank-prefixed; lower numbers are more similar to the retrieval query. Search and read it with local tools.`
									: `Workspace expanded under ./pull_${pullIndex}. Documents are directly inside that folder; search and read it with local tools.`
								: materializationMode !== "original"
									? `Workspace expanded under ./pull_${pullIndex}. Documents inside query folders are rank-prefixed; lower numbers are more similar to the retrieval query. Search and read that workspace with local tools.`
									: `Workspace expanded under ./pull_${pullIndex}. Search and read that workspace with local tools.`,
					},
				],
				details,
			};
		},

		renderCall(args, theme) {
			const text = new Text("", 0, 0);
			text.setText(formatPullCall(args, theme, toolName));
			return text;
		},

		renderResult(result, options, theme) {
			const text = new Text("", 0, 0);
			text.setText(formatPullResult(result, options, theme));
			return text;
		},
	};
}

export function createPullTool(cwd: string, options?: PullToolOptions): AgentTool<typeof pullSchema> {
	return wrapToolDefinition(createPullToolDefinition(cwd, options));
}
