# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os


def modify_yaml_file(file_path):
    updated = {
        "updated_compile": False,
        "updated_packed": False,
        "added_compile": False,
        "added_activation_offloading": False,
        "added_packed": False,
        "added_profiler": False,
        "updated_gradient_accumulation_steps": False,
        "updated_checkpointing_comment": False,
        "updated_gradient_comment": False,
        "updated_compile_comment": False,
        "updated_packed_comment": False,
    }

    with open(file_path, "r") as file:
        lines = file.readlines()
    # Step 2: Remove duplicate 'compile' entries
    compile_indices = [
        i for i, line in enumerate(lines) if line.strip().startswith("compile:")
    ]
    if len(compile_indices) > 1:
        for index in sorted(compile_indices, reverse=True):
            del lines[index]
        updated["updated_compile"] = True
    # Step 3: Move 'packed' after '_component_' and align indentation
    for i, line in enumerate(lines):
        if (
            line.strip().startswith("packed:")
            and i + 1 < len(lines)
            and "_component_" in lines[i + 1]
        ):
            packed_line = lines.pop(i)
            lines.insert(i + 1, packed_line)  # Insert after the _component_ line
            updated["updated_packed"] = True
            break
    # Step 4: Add 'compile' if missing
    if not any(line.strip().startswith("compile:") for line in lines):
        for i, line in enumerate(lines):
            if line.strip().startswith("max_steps_per_epoch:"):
                indentation = len(line) - len(line.lstrip())
                new_line = (
                    " " * indentation
                    + "compile: False # pytorch compile, set to true for better perf/memory\n"
                )
                lines.insert(i + 1, new_line)
                updated["added_compile"] = True
                break
    # Step 5: Add 'enable_activation_offloading' if missing
    if (
        not any(
            line.strip().startswith("enable_activation_offloading:") for line in lines
        )
        and "vision" not in file_path
        and "ppo" not in file_path
        and "dpo" not in file_path
        and "distillation" not in file_path
        and "qat" not in file_path
    ):
        for i, line in enumerate(lines):
            if line.strip().startswith("enable_activation_checkpointing:"):
                indentation = len(line) - len(line.lstrip())
                new_line = (
                    " " * indentation
                    + "enable_activation_offloading: False  # True reduces memory\n"
                )
                lines.insert(i + 1, new_line)
                updated["added_activation_offloading"] = True
                break
    # Step 6: Add 'packed' if missing
    if "dpo" not in file_path and "ppo" not in file_path:
        if (
            not any(line.strip().startswith("packed:") for line in lines)
            and "vision" not in file_path
        ):
            for i, line in enumerate(lines):
                if "_component_" in line and "dataset" in lines[i - 1]:
                    indentation = len(line) - len(line.lstrip())
                    new_line = (
                        " " * indentation + "packed: False # True increases speed\n"
                    )
                    lines.insert(i + 1, new_line)
                    updated["added_packed"] = True
                    break

    # Step 7: Replace/Add 'profiler' section if missing
    if "ppo" not in file_path and "dpo" not in file_path:
        profiler_section = """# Profiler (disabled)
profiler:
    _component_: torchtune.training.setup_torch_profiler
    enabled: False

    #Output directory of trace artifacts
    output_dir: ${output_dir}/profiling_outputs

    #`torch.profiler.ProfilerActivity` types to trace
    cpu: True
    cuda: True

    #trace options passed to `torch.profiler.profile`
    profile_memory: False
    with_stack: False
    record_shapes: True
    with_flops: False

    # `torch.profiler.schedule` options:
    # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
    wait_steps: 5
    warmup_steps: 3
    active_steps: 2
    num_cycles: 1
"""

        # Correct the 'profiler' section if it has incorrect indentation
        start_index = None
        end_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("# Profiler (disabled)"):
                start_index = i
            if line.strip().startswith("num_cycles: 1"):
                end_index = i
                break

        if start_index is not None and end_index is not None:
            # Replaces profiler

            # Remove the old section
            del lines[start_index : end_index + 1]
            # Insert the new section
            lines.insert(start_index, profiler_section)
            updated["added_profiler"] = True

        if not any(line.strip().startswith("profiler:") for line in lines):
            lines.append(profiler_section)
            updated["added_profiler"] = True

    # Step 8: Update 'gradient_accumulation_steps' if greater than 1
    for i, line in enumerate(lines):
        if line.strip().startswith("gradient_accumulation_steps:"):
            parts = line.split(":")
            if len(parts) > 1 and int(parts[1].strip()) > 1:
                lines[i] = parts[0] + ": 8\n"
                updated["updated_gradient_accumulation_steps"] = True
            break

    # Step 9: Add or replace comment for 'enable_activation_checkpointing'
    for i, line in enumerate(lines):
        if line.strip().startswith("enable_activation_checkpointing:"):
            parts = line.split("#")
            lines[i] = parts[0].strip() + "  # True reduces memory\n"
            updated["updated_checkpointing_comment"] = True
            break
    # Step 9.5: Add or replace comment for 'enable_activation_offloading'
    for i, line in enumerate(lines):
        if line.strip().startswith("enable_activation_offloading:"):
            parts = line.split("#")
            lines[i] = parts[0].strip() + "  # True reduces memory\n"
            updated["updated_checkpointing_comment"] = True
            break
    # Step 10: Add or replace comment for 'gradient_accumulation_steps'
    for i, line in enumerate(lines):
        if line.strip().startswith("gradient_accumulation_steps:"):
            parts = line.split("#")
            lines[i] = parts[0].rstrip() + "  # Use to increase virtual batch size\n"
            updated["updated_gradient_comment"] = True
            break
    # Step 11: Add or replace comment for 'compile'
    for i, line in enumerate(lines):
        if line.strip().startswith("compile:"):
            parts = line.split("#")
            lines[i] = (
                parts[0].rstrip()
                + "  # pytorch compile, set to true for better perf/memory\n"
            )
            updated["updated_compile_comment"] = True
            break
    # Step 12: Add or replace comment for 'packed'
    for i, line in enumerate(lines):
        if line.strip().startswith("packed:"):
            parts = line.split("#")
            lines[i] = parts[0].rstrip() + "  # True increases speed\n"
            updated["updated_packed_comment"] = True
            break

    # for files ending with "full.yaml" or "full_single_device.yaml"
    if (
        file_path.endswith("full.yaml")
        or file_path.endswith("full_single_device.yaml")
        and "qat" not in file_path
        and "ppo" not in file_path
        and "dpo" not in file_path
    ):
        # Step 13: Add 'optimizer_in_bwd: False' if missing
        if not any(line.strip().startswith("optimizer_in_bwd:") for line in lines):
            for i, line in enumerate(lines):
                if line.strip().startswith("compile:"):
                    indentation = len(line) - len(line.lstrip())
                    new_line = " " * indentation + "optimizer_in_bwd: False\n"
                    lines.insert(i + 1, new_line)
                    updated["added_optimizer_in_bwd"] = True
                    break

    # Step 14: Add/replace comment for 'optimizer_in_bwd'
    for i, line in enumerate(lines):
        if line.strip().startswith("optimizer_in_bwd:"):
            parts = line.split("#")
            lines[i] = (
                parts[0].rstrip()
                + "  # True saves memory. Requires gradient_accumulation_steps=1\n"
            )
            updated["updated_optimizer_in_bwd_comment"] = True
            break

    # Step 14.5: Add/replace comment for 'custom_sharded_layers'
    for i, line in enumerate(lines):
        if line.strip().startswith("custom_sharded_layers:"):
            parts = line.split("#")
            lines[i] = (
                parts[0].rstrip()
                + "  # Shard it separetely. Useful for large vocab size. Lower Memory, but lower speed\n"
            )
            updated["updated_custom_sharded_layers_comment"] = True
            break

    # for files with lora in the name
    if "lora" in file_path or "dora" in file_path:
        for i, line in enumerate(lines):
            # Step 15: make 'lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']'
            if line.strip().startswith("lora_attn_modules:"):
                lines[i] = "  lora_attn_modules: ['q_proj', 'v_proj', 'output_proj']\n"
            # Step 16: make 'apply_lora_to_mlp: True'
            elif line.strip().startswith("apply_lora_to_mlp:"):
                lines[i] = "  apply_lora_to_mlp: True\n"
            # Step 17: add comment to 'lora_rank'
            elif line.strip().startswith("lora_rank:"):
                parts = line.split("#")
                lines[i] = (
                    parts[0].rstrip() + "  # higher increases accuracy and memory\n"
                )
            # Step 18: add comment to 'lora_alpha'
            elif line.strip().startswith("lora_alpha:"):
                parts = line.split("#")
                lines[i] = parts[0].rstrip() + "  # usually alpha=2*rank\n"
    with open(file_path, "w") as file:
        file.writelines(lines)
    return updated


def search_yaml_files(directory):
    updated_files = []
    not_updated_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "_" not in file or "generation" in file or "evaluation" in file:
                print(f"Skipping {file}")
                continue
            file_path = os.path.join(root, file)
            updates = modify_yaml_file(file_path)
            if any(updates.values()):
                updated_files.append({file_path: updates})
            else:
                not_updated_files.append(file_path)
    print("Updated files and changes:")
    for update in updated_files:
        print(update)
    print("\nFiles not updated:")
    for file in not_updated_files:
        print(file)


directory = "recipes/configs"
search_yaml_files(directory)
