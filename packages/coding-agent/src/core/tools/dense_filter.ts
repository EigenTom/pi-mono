/*
	Document Dense Filter Tool
	Let Agents provide multiple sub queries (e,g, m queries), call the backend dense-retriever to filter n documents, simplify the large corpus into a subset comprising of at most (m*n) documents
	Support 1) file hard link 2) manifest doc for document isolation.
*/

import { link, mkdir, readFile, unlink, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, join, normalize, relative, resolve } from "node:path";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import type { TextContent } from "@mariozechner/pi-ai";
import { Text } from "@mariozechner/pi-tui";
import { type Static, Type } from "@sinclair/typebox";
import fetch, { type RequestInit, type Response } from "node-fetch";
import type { ToolDefinition, ToolRenderResultOptions } from "../extensions/types.js";
import { getTextOutput } from "./render-utils.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";

// TODO: ablation on maxItems param
const denseFilterSchema = Type.Object({
	queries: Type.Array(Type.String({ minLength: 1 }), {
		minItems: 1,
		maxItems: 8,
		description: "Focused semantic sub-queries used to identify documents that should not be excluded.",
	}),
});

export type DenseFilterToolInput = Static<typeof denseFilterSchema>;

type RetrieverResult = {
	docid?: string;
	doc_path: string;
	score: number;
};

type FilteredDocument = {
	docid?: string;
	doc_path: string;
	score: number;
	queries: string[];
};

export interface DenseFilterToolDetails {
	queries: string[];
	topKPerQuery: number;
	maxDocuments: number;
	viewMode: DenseFilterViewMode;
	viewDir: string;
	manifestPath?: string;
	managedPathsPath?: string;
	visibleDocumentCount: number;
	perQueryHitCounts: Record<string, number>;
	materialized?: {
		createdCount: number;
		missingCount: number;
	};
}

// by default we use hardlink. manifest mode (white list mode) needs extra implementation at bash/grep tools' harness level
export type DenseFilterViewMode = "manifest" | "hardlink";

export interface DenseFilterOperations {
	fetch: (url: string, options: RequestInit) => Promise<Response>;
	writeFile: (path: string, content: string) => Promise<void>;
	readFile: (path: string) => Promise<string>;
	mkdir: (path: string) => Promise<void>;
	link: (existingPath: string, newPath: string) => Promise<void>;
	unlink: (path: string) => Promise<void>;
}

const defaultDenseFilterOperations: DenseFilterOperations = {
	fetch: (url, options) => fetch(url, options),
	writeFile: (path, content) => writeFile(path, content, "utf-8"),
	readFile: (path) => readFile(path, "utf-8"),
	mkdir: (path) => mkdir(path, { recursive: true }).then(() => {}),
	link: (existingPath, newPath) => link(existingPath, newPath),
	unlink: (path) => unlink(path),
};

export interface DenseFilterToolOptions {
	baseUrl?: string;
	topKPerQuery?: number;
	maxDocuments?: number;
	viewMode?: DenseFilterViewMode;
	viewDir?: string;
	sourceRoot?: string;
	toolName?: string;
	operations?: Partial<DenseFilterOperations>;
}

const FILTER_DIR = ".dci_filter";
const MANIFEST_FILE = "manifest.json";
const MANAGED_PATHS_FILE = "managed_paths.json";

function readPositiveIntEnv(name: string, fallback: number): number {
	const raw = process.env[name];
	if (!raw) return fallback;
	const value = Number.parseInt(raw, 10);
	return Number.isFinite(value) && value > 0 ? value : fallback;
}

function resolveViewMode(options?: DenseFilterToolOptions): DenseFilterViewMode {
	// The harness controls view mode through env/options. The model only sees
	// the simple { queries } schema, so benchmark knobs do not become prompt load.
	const raw = options?.viewMode ?? process.env.DCI_DENSE_FILTER_VIEW_MODE ?? "manifest";
	return raw === "hardlink" ? "hardlink" : "manifest";
}

function safeRelativePath(docPath: string): string | undefined {
	// Retriever doc_path values are expected to be corpus-relative. Reject absolute
	// and parent-traversal paths before using them for manifest or hardlink targets.
	const normalized = normalize(docPath.replace(/\\/g, "/"));
	if (!normalized || normalized === "." || isAbsolute(normalized)) return undefined;
	if (normalized === ".." || normalized.startsWith(`..`)) return undefined;
	return normalized;
}

function isInside(parent: string, child: string): boolean {
	const rel = relative(parent, child);
	return rel === "" || (!!rel && !rel.startsWith("..") && !isAbsolute(rel));
}

async function readJsonArray(path: string, ops: DenseFilterOperations): Promise<string[]> {
	try {
		const text = await ops.readFile(path);
		const parsed = JSON.parse(text);
		return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
	} catch {
		return [];
	}
}

// merge multiple queries' related documents that is retrieved from local dense retriever into a big corpus without duplication
function mergeResults(queries: string[], perQueryHits: Record<string, RetrieverResult[]>, maxDocuments: number) {
	const merged = new Map<string, FilteredDocument>();

	for (const query of queries) {
		for (const hit of perQueryHits[query] ?? []) {
			const key = hit.docid ? `docid:${hit.docid}` : `path:${hit.doc_path}`;
			const existing = merged.get(key);
			if (!existing) {
				merged.set(key, {
					docid: hit.docid,
					doc_path: hit.doc_path,
					score: hit.score,
					queries: [query],
				});
			} else {
				existing.score = Math.max(existing.score, hit.score);
				if (!existing.queries.includes(query)) existing.queries.push(query);
			}
		}
	}

	return Array.from(merged.values())
		.sort((a, b) => b.score - a.score || a.doc_path.localeCompare(b.doc_path))
		.slice(0, maxDocuments);
}

// retrieve the related document list for ONE query
async function retrieveOne(
	baseUrl: string,
	query: string,
	topK: number,
	ops: DenseFilterOperations,
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
		throw new Error(`Dense Filter API error: ${response.status} ${response.statusText} - ${text}`);
	}
	const data = (await response.json()) as { results?: RetrieverResult[] };
	return Array.isArray(data.results) ? data.results : [];
}

async function materializeHardlinks(args: {
	documents: FilteredDocument[];
	viewDir: string;
	sourceRoot: string;
	filterDir: string;
	ops: DenseFilterOperations;
}): Promise<{ created: string[]; missing: string[] }> {
	const { documents, viewDir, sourceRoot, filterDir, ops } = args;
	// Keep the writable view outside the immutable source corpus. This prevents a
	// stale cleanup pass from unlinking original corpus files by accident.
	if (isInside(sourceRoot, viewDir) || isInside(viewDir, sourceRoot)) {
		throw new Error("Hardlink viewDir must be separate from DCI_DENSE_FILTER_SOURCE_ROOT");
	}

	await ops.mkdir(filterDir);
	const managedPath = join(filterDir, MANAGED_PATHS_FILE);
	const previous = await readJsonArray(managedPath, ops);
	// Only remove paths that this tool previously created in the view. User-created
	// files, logs, and .dci_filter metadata are left alone.
	for (const relPath of previous) {
		const safePath = safeRelativePath(relPath);
		if (!safePath) continue;
		try {
			await ops.unlink(join(viewDir, safePath));
		} catch {
			// Ignore stale managed entries.
		}
	}

	const created: string[] = [];
	const missing: string[] = [];
	for (const doc of documents) {
		const safePath = safeRelativePath(doc.doc_path);
		if (!safePath) {
			missing.push(doc.doc_path);
			continue;
		}

		const sourcePath = resolve(sourceRoot, safePath);
		if (!isInside(sourceRoot, sourcePath)) {
			missing.push(doc.doc_path);
			continue;
		}

		const targetPath = resolve(viewDir, safePath);
		if (!isInside(viewDir, targetPath)) {
			missing.push(doc.doc_path);
			continue;
		}

		try {
			await ops.mkdir(dirname(targetPath));
			try {
				await ops.unlink(targetPath);
			} catch {
				// Target may not exist.
			}
			// Hardlinks create new directory entries pointing at the same file content.
			// They give bash/rg/read a normal file tree without copying document bytes.
			await ops.link(sourcePath, targetPath);
			created.push(safePath);
		} catch {
			missing.push(doc.doc_path);
		}
	}

	await ops.writeFile(managedPath, JSON.stringify(created, null, 2));
	return { created, missing };
}

function formatDenseFilterCall(
	args: { queries?: string[] } | undefined,
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
): string {
	const count = args?.queries?.length ?? 0;
	const label = count === 1 ? "1 query" : `${count} queries`;
	return `${theme.fg("toolTitle", theme.bold("filter"))} ${theme.fg("toolOutput", label)}`;
}

function formatDenseFilterResult(
	result: { content: TextContent[] },
	_options: ToolRenderResultOptions,
	theme: typeof import("../../modes/interactive/theme/theme.js").theme,
): string {
	return `\n${theme.fg("toolOutput", getTextOutput(result, false))}`;
}

export function createDenseFilterToolDefinition(
	cwd: string,
	options?: DenseFilterToolOptions,
): ToolDefinition<typeof denseFilterSchema, DenseFilterToolDetails> {
	// Runtime configuration is intentionally outside the tool schema: the agent
	// chooses semantic queries, while the harness chooses retriever URL, topK,
	// candidate cap, source root, and whether to materialize a filesystem view.
	const baseUrl = options?.baseUrl ?? process.env.DCI_DENSE_FILTER_BASE_URL ?? "http://localhost:8000/retrieve";
	const topKPerQuery = options?.topKPerQuery ?? readPositiveIntEnv("DCI_DENSE_FILTER_TOP_K", 100);
	const maxDocuments = options?.maxDocuments ?? readPositiveIntEnv("DCI_DENSE_FILTER_MAX_DOCS", 500);
	const viewMode = resolveViewMode(options);
	const viewDir = resolve(cwd, options?.viewDir ?? process.env.DCI_DENSE_FILTER_VIEW_DIR ?? ".");
	const sourceRoot = options?.sourceRoot ?? process.env.DCI_DENSE_FILTER_SOURCE_ROOT;
	const toolName = options?.toolName ?? "dense_filter";
	const ops = { ...defaultDenseFilterOperations, ...options?.operations };

	return {
		name: toolName,
		label: "Filter Corpus",
		description:
			"Remove most semantically unrelated documents from the current benchmark corpus using multiple semantic sub-queries. Updates the current corpus directory and returns only a short status.",
		promptSnippet:
			"filter removes most documents that are semantically unrelated to the question from the current corpus directory. After using it, continue normal local search and evidence reading with bash, rg, find, ls, and read.",
		promptGuidelines: [
			"Use filter once near the beginning after decomposing a document-retrieval question into focused semantic sub-queries.",
			"The filter operation removes most semantically unrelated documents from the visible working corpus; it does not answer the question or rank evidence.",
			"After filter, continue normal DCI-style local exploration with bash, rg, find, ls, and read, starting from rare clue anchors.",
			// "After filter, treat the current corpus as the corpus and continue ordinary local exploration with shell commands such as bash, rg, etc.",
			// "Do not answer from the filter status alone; answer only after reading supporting document text.",
		],
		parameters: denseFilterSchema,

		async execute(_toolCallId, params: DenseFilterToolInput, signal?: AbortSignal) {
			// Deduplicate after trimming so repeated paraphrases do not waste retriever calls.
			const queries = Array.from(new Set(params.queries.map((query) => query.trim()).filter(Boolean)));
			if (queries.length === 0) {
				throw new Error("At least one non-empty query is required");
			}

			const perQueryHits: Record<string, RetrieverResult[]> = {};
			for (const query of queries) {
				if (signal?.aborted) throw new Error("Operation aborted");
				// The retriever endpoint returns ranked documents for one semantic query.
				// mergeResults below unions and caps the multi-query result set.
				perQueryHits[query] = await retrieveOne(baseUrl, query, topKPerQuery, ops, signal);
			}

			const visibleDocuments = mergeResults(queries, perQueryHits, maxDocuments);
			const filterDir = join(viewDir, FILTER_DIR);
			const manifestPath = viewMode === "manifest" ? join(filterDir, MANIFEST_FILE) : undefined;
			const managedPathsPath = join(filterDir, MANAGED_PATHS_FILE);

			let materialized: Awaited<ReturnType<typeof materializeHardlinks>> | undefined;
			if (viewMode === "hardlink") {
				if (!sourceRoot) {
					throw new Error("DCI_DENSE_FILTER_SOURCE_ROOT is required when DCI_DENSE_FILTER_VIEW_MODE=hardlink");
				}
				// In hardlink mode, cwd/viewDir becomes the visible corpus. Standard
				// bash/rg/read naturally operate only on the materialized candidates.
				materialized = await materializeHardlinks({
					documents: visibleDocuments,
					viewDir,
					sourceRoot: resolve(sourceRoot),
					filterDir,
					ops,
				});
			}

			const perQueryHitCounts = Object.fromEntries(
				Object.entries(perQueryHits).map(([query, hits]) => [query, hits.length]),
			);
			// Keep result.details compact because pi records it in state/events. The
			// hardlink view is the source of truth in hardlink mode; manifest mode is
			// the only mode that writes the full candidate JSON.
			const details: DenseFilterToolDetails = {
				queries,
				topKPerQuery,
				maxDocuments,
				viewMode,
				viewDir,
				manifestPath,
				managedPathsPath: viewMode === "hardlink" ? managedPathsPath : undefined,
				visibleDocumentCount: visibleDocuments.length,
				perQueryHitCounts,
				materialized: materialized
					? {
							createdCount: materialized.created.length,
							missingCount: materialized.missing.length,
						}
					: undefined,
			};
			if (viewMode === "manifest" && manifestPath) {
				await ops.mkdir(filterDir);
				await ops.writeFile(
					manifestPath,
					JSON.stringify(
						{
							created_at: new Date().toISOString(),
							queries,
							top_k_per_query: topKPerQuery,
							max_documents: maxDocuments,
							view_mode: viewMode,
							view_dir: viewDir,
							visible_documents: visibleDocuments,
							per_query_hits: perQueryHits,
							materialized,
						},
						null,
						2,
					),
				);
			}

			return {
				content: [
					{
						type: "text",
						text:
							viewMode === "hardlink"
								? "Corpus has been slimmed down. Continue searching the current corpus with bash, rg, find, ls, and read."
								: "Corpus filter state has been updated. Continue searching the current corpus with bash, rg, find, ls, and read.",
					},
				],
				details,
			};
		},

		renderCall(args, theme) {
			const text = new Text("", 0, 0);
			text.setText(formatDenseFilterCall(args, theme));
			return text;
		},

		renderResult(result, options, theme) {
			const text = new Text("", 0, 0);
			text.setText(formatDenseFilterResult(result as any, options, theme));
			return text;
		},
	};
}

export function createDenseFilterTool(
	cwd: string,
	options?: DenseFilterToolOptions,
): AgentTool<typeof denseFilterSchema> {
	return wrapToolDefinition(createDenseFilterToolDefinition(cwd, options));
}
