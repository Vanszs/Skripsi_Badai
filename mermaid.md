flowchart LR
    A((Mulai)) --> B[Studi Literatur]
    B --> C[Perancangan Model]
    C --> D[Akuisisi Data ERA5]
    D --> E[Data Preprocessing]
    E --> F[Pembagian Data Temporal]
    F --> G[Pelatihan RA-Diffusion + ST-GNN]
    G --> H[Inference dan Ensemble Sampling]
    H --> I[Pengujian dan Evaluasi]
    I --> J[Perbandingan Metrik]
    J --> K((Selesai))

    classDef terminal fill:#d9ead3,stroke:#6aa84f,stroke-width:1.5px,color:#1f1f1f;
    classDef process fill:#d9d2e9,stroke:#8e7cc3,stroke-width:1.2px,color:#1f1f1f;
    class A,K terminal;
    class B,C,D,E,F,G,H,I,J process;

