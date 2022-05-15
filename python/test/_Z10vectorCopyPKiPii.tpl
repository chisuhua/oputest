	.text
	.global {{kname}}
	.type {{kname}},@function
{{kname}}:
bb_00:
		v_load_u64	v[3:4],	{{kname}}_param_0 % mspace:param
		v_load_u64	v[5:6],	{{kname}}_param_1 % mspace:param
		v_load_u32	v7,	{{kname}}_param_2 % mspace:param
bb_01:
		v_mov_b32	v8,	block_dim_x
		v_mov_b32	v9,	block_idx_x
		v_mullo_i32_i32	v10,	v8,	v9
		v_mov_b32	v11,	thread_idx_x
		v_add_i32	v12,	v10,	v11
bb_02:
		v_cmp_lt_i32	tcc0,	v12,	v7
		s_not_b32	tcc1,	tcc0
		s_cbranch_tccz tcc1,	bb_exit
bb_03:
		s_branch  bb_04
bb_04:
		v_sext_i64_i32	v[13:14],	v12
		v_lshl_b64	v[15:16],	v[13:14],	0x2
		v_add_i64	v[17:18],	v[3:4],	v[15:16]
		v_load_u32	v19,	v[17:18] % mspace:flat
{{test_code}}
		v_sext_i64_i32	v[20:21],	v12
		v_lshl_b64	v[22:23],	v[20:21],	0x2
		v_add_i64	v[24:25],	v[5:6],	v[22:23]
		v_store_u32	v19,	v[24:25] % mspace:flat
bb_exit:
		t_exit

---
opu.kernels:
  - .args:
      - .address_space: global
        .name: {{kname}}_param_0
        .offset: 0
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: {{kname}}_param_1
        .offset: 8
        .size: 8
        .value_kind: global_buffer
      - .address_space: global
        .name: {{kname}}_param_2
        .offset: 16
        .size: 4
        .value_kind: global_buffer
    .name: {{kname}}
    .local_framesize: 0
    .smem: 0
    .lmem: 0
    .cmem: 372
    .kernel_ctrl: 69761
    .kernel_mode: 0
    .bar_used: 0
opu.version:
  - 2
  - 0
...
