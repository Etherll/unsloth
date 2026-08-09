// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerStoreStubResolver,
} from "./helpers/kit.ts";

registerStoreStubResolver();
installLocalStorageFake();
const { buildTrainingStartPayload } = await import(
  "../src/features/training/api/mappers.ts"
);
const { initialTrainingConfigState } = await import(
  "../src/features/training/stores/training-config-policy.ts"
);
const { parseYamlConfig, serializeConfigToYaml } = await import(
  "../src/features/training/lib/yaml-config.ts"
);
const { mapBackendModelConfigToTrainingPatch } = await import(
  "../src/features/training/lib/model-defaults.ts"
);

const TEMPLATE =
  "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\\n{% endfor %}";

function config(overrides = {}) {
  return {
    ...initialTrainingConfigState,
    selectedModel: "Qwen/Qwen3.5-0.8B-Base",
    modelType: "text" as const,
    dataset: "org/conversations",
    datasetSplit: "train",
    chatTemplate: TEMPLATE,
    ...overrides,
  };
}

test("conversational SFT sends the custom chat template unchanged", () => {
  const payload = buildTrainingStartPayload(config(), null);
  assert.equal(payload.chat_template, TEMPLATE);
});

test("raw, CPT, and embedding training never send a chat template", () => {
  assert.equal(
    buildTrainingStartPayload(config({ datasetFormat: "raw" }), null)
      .chat_template,
    null,
  );
  assert.equal(
    buildTrainingStartPayload(config({ trainingMethod: "cpt" }), null)
      .chat_template,
    null,
  );
  assert.equal(
    buildTrainingStartPayload(config({ isEmbeddingModel: true }), null)
      .chat_template,
    null,
  );
});

test("YAML save and load preserves multiline Jinja exactly", () => {
  const saved = serializeConfigToYaml(config(), false);
  const patch = mapBackendModelConfigToTrainingPatch(parseYamlConfig(saved));
  assert.equal(patch.chatTemplate, TEMPLATE);
});
