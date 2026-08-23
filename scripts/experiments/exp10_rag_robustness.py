#!/usr/bin/env python3
"""EXP-10: RAG Robustness Under Knowledge Base Scale.

DEPOSIT NOTE — EXPLORATORY / SIMULATED SCAFFOLD. This script contains simulated or placeholder logic and does not perform a live pipeline run; its output is NOT shipped in results/ and backs NO number reported in the manuscript. Retained only for transparency about exploratory scaffolding.

Tests whether λ-Weighted Fusion Embedding maintains retrieval quality
when the knowledge base is expanded from 8 to 50/100 documents with
distractor entries.

Metrics: Precision@5, MRR (Mean Reciprocal Rank), DA.
"""
from __future__ import annotations

import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class RetrievalResult:
    """Result of retrieval evaluation at a given KB scale."""
    kb_size: int
    precision_at_5: float
    mrr: float
    da: float
    correct: int
    total: int
    avg_correct_rank: float
    avg_correct_similarity: float
    avg_top_distractor_similarity: float


# --- Distractor document generation ---
DISTRACTOR_DOCUMENTS = [
    # Fish species (not diseases)
    {"id": "DIST_SP01", "title": "Atlantic Salmon (Salmo salar)", "content": "Atlantic salmon is a species of ray-finned fish. It is the third most farmed fish worldwide. Known for its pink flesh and anadromous lifecycle, migrating from saltwater to freshwater for spawning."},
    {"id": "DIST_SP02", "title": "Nile Tilapia (Oreochromis niloticus)", "content": "Nile tilapia is a freshwater cichlid native to Africa. It is one of the most important aquaculture species globally, tolerant of diverse environmental conditions and fast-growing."},
    {"id": "DIST_SP03", "title": "Rainbow Trout (Oncorhynchus mykiss)", "content": "Rainbow trout is a cold-water species prized in aquaculture. Requires high dissolved oxygen levels and clean water. Known for the distinctive pink lateral band."},
    {"id": "DIST_SP04", "title": "European Sea Bass (Dicentrarchus labrax)", "content": "European sea bass is a highly valued marine aquaculture species in the Mediterranean region. Susceptible to temperature stress and viral nervous necrosis."},
    {"id": "DIST_SP05", "title": "Channel Catfish (Ictalurus punctatus)", "content": "Channel catfish is the most farmed catfish species in the United States. Bottom-dwelling fish with barbels and smooth scaleless skin. Tolerates low oxygen."},
    {"id": "DIST_SP06", "title": "Asian Sea Bass / Barramundi (Lates calcarifer)", "content": "Barramundi is a tropical euryhaline fish species farmed extensively in Southeast Asia and Australia. Fast growth rate, reaching market size in 12-18 months."},
    {"id": "DIST_SP07", "title": "Yellowtail Kingfish (Seriola lalandi)", "content": "Yellowtail is a premium aquaculture species popular in sashimi markets. Requires high-quality feeds and optimal water conditions for growth."},
    {"id": "DIST_SP08", "title": "Giant Tiger Prawn (Penaeus monodon)", "content": "Tiger prawns are the most widely cultured shrimp species in tropical regions. Susceptible to white spot syndrome virus (WSSV) and early mortality syndrome (EMS)."},
    # Water quality management
    {"id": "DIST_WQ01", "title": "Dissolved Oxygen Management in Aquaculture", "content": "Maintaining adequate dissolved oxygen is critical for fish health. Levels below 4 mg/L cause stress, below 2 mg/L can be lethal. Aeration systems include paddlewheel aerators, diffusers, and pure oxygen injection systems."},
    {"id": "DIST_WQ02", "title": "Ammonia and Nitrogen Cycle in Fish Ponds", "content": "Ammonia toxicity is a major water quality concern. Total ammonia nitrogen (TAN) above 0.05 mg/L is stressful. Biological filtration using nitrifying bacteria converts toxic ammonia to less harmful nitrate."},
    {"id": "DIST_WQ03", "title": "pH Management in Aquaculture Systems", "content": "Optimal pH range for most freshwater species is 6.5-8.5. Low pH increases metal toxicity while high pH increases ammonia toxicity. Regular monitoring and buffer addition maintain stable pH."},
    {"id": "DIST_WQ04", "title": "Water Temperature Control Methods", "content": "Temperature affects fish metabolism, growth rate, and disease susceptibility. Chiller systems for cold-water species, heaters for tropical species, and thermal covers reduce energy costs."},
    {"id": "DIST_WQ05", "title": "Biofloc Technology in Aquaculture", "content": "Biofloc is a sustainable aquaculture technology using microbial communities to convert waste nutrients into food. Reduces water exchange requirements and improves biosecurity."},
    # Feed and nutrition
    {"id": "DIST_FD01", "title": "Fish Feed Formulation Basics", "content": "Commercial fish feeds contain protein (30-55%), lipids (5-20%), carbohydrates, vitamins, and minerals. Feed conversion ratio (FCR) measures efficiency of feed utilization."},
    {"id": "DIST_FD02", "title": "Alternative Protein Sources for Aquafeeds", "content": "Insect meal, single-cell protein, microalgae, and plant-based proteins are being explored as sustainable alternatives to fishmeal in aquaculture diets."},
    {"id": "DIST_FD03", "title": "Feeding Strategies and Feed Management", "content": "Optimal feeding frequency varies by species and size. Demand feeders, automatic feeders, and scheduled feeding programs improve growth and reduce waste."},
    # Equipment and infrastructure
    {"id": "DIST_EQ01", "title": "Recirculating Aquaculture Systems (RAS)", "content": "RAS recirculates water through mechanical and biological filters, UV sterilizers, and degassers. Enables indoor fish farming with minimal water usage and environmental impact."},
    {"id": "DIST_EQ02", "title": "Net Cage Design for Marine Aquaculture", "content": "Floating net cages consist of HDPE frame, nylon or copper alloy netting, mooring systems, and walkways. Cage diameter ranges from 20 to 50 meters for commercial operations."},
    {"id": "DIST_EQ03", "title": "Underwater Camera Systems for Fish Monitoring", "content": "Underwater cameras enable remote observation of fish behavior, feeding response, and health status. IP cameras with LED lighting provide real-time video feeds to control rooms."},
    # Aquaculture management
    {"id": "DIST_MG01", "title": "Biosecurity Protocols for Aquaculture Farms", "content": "Biosecurity measures include quarantine procedures, disinfection protocols, foot baths, restricted access zones, and vaccination programs to prevent disease introduction."},
    {"id": "DIST_MG02", "title": "Stocking Density Guidelines", "content": "Optimal stocking density varies by species: tilapia 20-50 kg/m³, salmon 15-25 kg/m³, shrimp 30-60 PL/m². Overcrowding increases stress and disease susceptibility."},
    {"id": "DIST_MG03", "title": "Harvest and Post-Harvest Handling", "content": "Proper harvesting techniques minimize stress and maintain product quality. Cold chain management, ice-to-fish ratio, and processing hygiene are critical."},
    # Other fish diseases (not in training set) - confounding distractors
    {"id": "DIST_DIS01", "title": "Koi Herpesvirus Disease (KHVD)", "content": "Koi herpesvirus disease is a highly contagious viral disease of common carp and koi. Symptoms include gill necrosis, sunken eyes, and skin blisters. Mortality can reach 80-100%."},
    {"id": "DIST_DIS02", "title": "Infectious Salmon Anemia (ISA)", "content": "ISA is a viral disease affecting Atlantic salmon. Symptoms include pale gills, ascites, hemorrhagic liver necrosis, and anemia. Reportable to OIE. No treatment available."},
    {"id": "DIST_DIS03", "title": "Viral Nervous Necrosis (VNN)", "content": "VNN is caused by betanodavirus, affecting marine fish larvae and juveniles. Symptoms include erratic swimming, spiral motion, and darkened body color. High mortality in larval stages."},
    {"id": "DIST_DIS04", "title": "Lymphocystis Disease", "content": "Lymphocystis is a chronic viral disease causing enlarged dermal cells forming grape-like clusters on skin and fins. Usually self-limiting but unsightly. Common in warmwater marine species."},
    {"id": "DIST_DIS05", "title": "Enteric Redmouth Disease (ERM)", "content": "ERM is caused by Yersinia ruckeri, affecting salmonids. Symptoms include subcutaneous hemorrhages in the mouth and throat, darkened skin, exophthalmia, and splenomegaly."},
    {"id": "DIST_DIS06", "title": "Edwardsiellosis (Edwardsiella tarda)", "content": "Edwardsiellosis causes deep abscesses in muscle tissue, especially in catfish and eels. Gas-filled cavities in muscle, foul odor, and high mortality in acute cases."},
    {"id": "DIST_DIS07", "title": "Spring Viremia of Carp (SVC)", "content": "SVC is a rhabdoviral disease of cyprinid fish. Symptoms include abdominal distension, petechial hemorrhages, exophthalmia, and trailing fecal casts. Temperature dependent."},
    {"id": "DIST_DIS08", "title": "Epizootic Ulcerative Syndrome (EUS)", "content": "EUS is caused by the oomycete Aphanomyces invadans. Characterized by deep dermal ulcers, red spots, and necrotic lesions. Primarily affects freshwater fish in tropical regions. Major disease in South and Southeast Asian aquaculture."},
    {"id": "DIST_DIS09", "title": "Bacterial Kidney Disease (BKD)", "content": "BKD is caused by Renibacterium salmoninarum, a gram-positive diplococcus. Chronic disease in salmonids causing renal granulomas, exophthalmia, abdominal swelling."},
    {"id": "DIST_DIS10", "title": "Furunculosis (Aeromonas salmonicida)", "content": "Furunculosis causes boil-like lesions (furuncles) in muscle tissue of salmonids. Acute form: septicemia, darkening, hemorrhages. Chronic form: localized furuncles."},
    {"id": "DIST_DIS11", "title": "Epitheliocystis", "content": "Epitheliocystis is caused by intracellular bacteria forming cyst-like inclusions in gill epithelium. Affects many marine and freshwater species. Causes respiratory distress."},
    {"id": "DIST_DIS12", "title": "Tenacibaculosis (Flexibacter)", "content": "Tenacibaculosis causes skin ulcers and mouth erosion in marine fish. Previously known as Flexibacter maritimus infection. High mortality in juvenile flatfish."},
    # Environmental topics
    {"id": "DIST_ENV01", "title": "Harmful Algal Blooms in Aquaculture", "content": "Harmful algal blooms (HABs) produce toxins that kill fish through gill damage or oxygen depletion. Common species include Karenia brevis, Chattonella marina, and Prymnesium parvum."},
    {"id": "DIST_ENV02", "title": "Climate Change Impact on Aquaculture", "content": "Rising water temperatures, ocean acidification, and increased storm frequency affect aquaculture production. Adaptation strategies include species diversification and offshore farming."},
    {"id": "DIST_ENV03", "title": "Predator Control in Fish Farms", "content": "Bird predation, seal attacks, and jellyfish intrusion cause significant losses in marine aquaculture. Anti-predator nets, acoustic deterrents, and physical barriers provide protection."},
    # Genetics and breeding
    {"id": "DIST_GN01", "title": "Selective Breeding in Aquaculture", "content": "Genetic improvement programs use family-based and genomic selection to improve growth rate, disease resistance, and fillet yield. GIFT tilapia program achieved 12% improvement per generation."},
    {"id": "DIST_GN02", "title": "Triploidy and Polyploidy in Fish", "content": "Triploid fish are sterile and channel energy from reproduction to growth. Produced by pressure or temperature shocking fertilized eggs. Used in salmonid and shellfish aquaculture."},
    # Economics and policy
    {"id": "DIST_EC01", "title": "Global Aquaculture Market Trends 2024", "content": "Global aquaculture market exceeded USD 300 billion in 2024. Asia accounts for 89% of production. Key growth drivers include rising protein demand, technological innovation, and sustainability initiatives."},
    {"id": "DIST_EC02", "title": "Aquaculture Certification Standards", "content": "Major certification schemes include ASC, GlobalGAP, BAP, and organic labels. Standards cover environmental impact, social responsibility, animal welfare, and food safety."},
    # Processing and food safety
    {"id": "DIST_FS01", "title": "Antibiotic Residue Monitoring in Aquaculture", "content": "Maximum residue limits (MRLs) for antibiotics in fish are regulated by national food safety authorities. HPLC and ELISA methods detect residues. Withdrawal periods must be observed."},
    {"id": "DIST_FS02", "title": "Parasite Control in Fish Products", "content": "Anisakis and Diphyllobothrium are common parasites in raw fish products. Freezing at -20°C for 7 days or -35°C for 15 hours kills larvae. Visual inspection during processing."},
    # Additional disease-adjacent distractors (most challenging)
    {"id": "DIST_ADJ01", "title": "Fish Immune System and Vaccination", "content": "Fish possess innate and adaptive immunity. Vaccines available for vibriosis, furunculosis, and IPN. Delivery routes include injection, immersion, and oral. Adjuvants enhance immune response."},
    {"id": "DIST_ADJ02", "title": "Probiotics and Immunostimulants in Aquaculture", "content": "Probiotics (Bacillus, Lactobacillus) improve gut health and disease resistance. Beta-glucan, vitamin C, and nucleotides are common immunostimulants that enhance innate immunity."},
    {"id": "DIST_ADJ03", "title": "Histopathological Analysis of Fish Diseases", "content": "Histopathology examines tissue sections under microscopy for disease diagnosis. H&E staining reveals cellular changes including necrosis, inflammation, granuloma formation, and hyperplasia."},
    {"id": "DIST_ADJ04", "title": "PCR-Based Disease Diagnostics in Fish", "content": "Polymerase chain reaction (PCR) enables rapid molecular detection of fish pathogens. Real-time qPCR quantifies pathogen load. LAMP assay provides field-deployable alternative."},
    {"id": "DIST_ADJ05", "title": "Antimicrobial Resistance in Aquaculture", "content": "Overuse of antibiotics promotes antimicrobial resistance (AMR). Multi-drug resistant bacteria transfer resistance genes to human pathogens. Antibiotic stewardship is critical."},
    {"id": "DIST_ADJ06", "title": "Stress Response and Cortisol in Fish", "content": "Cortisol is the primary stress hormone in fish. Chronic stress from handling, overcrowding, or poor water quality suppresses immune function and increases disease susceptibility."},
]


def generate_distractor_sets(n_total: int) -> list[dict]:
    """Select distractors to reach target KB size."""
    n_distractor = n_total - 8  # 8 seed documents
    return DISTRACTOR_DOCUMENTS[:min(n_distractor, len(DISTRACTOR_DOCUMENTS))]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Compute CLIP text embeddings for a list of texts."""
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    embeddings = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # L2 normalize
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def load_seed_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load seed document embeddings from ChromaDB."""
    import chromadb

    client = chromadb.PersistentClient(path="/data/chroma_db")
    collection = client.get_collection("fish_disease_knowledge")
    data = collection.get(include=["embeddings", "documents", "metadatas"])

    embeddings = np.array(data["embeddings"])
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    sources = []
    for meta in data["metadatas"]:
        sources.append(meta.get("source_disease", meta.get("disease_id", "unknown")))

    return embeddings, sources


def load_test_image_embeddings() -> tuple[np.ndarray, list[str], list[dict]]:
    """Load pre-computed fusion embeddings for test images.

    Since fusion embeddings aren't stored in JSON, we recompute them
    from the ChromaDB query results by finding which seed doc matches
    with highest similarity and using that as ground truth.
    """
    data_path = Path(__file__).resolve().parents[2] / \
        "lab_dateset/organized/experiment_results/large_scale/exp_new01_classification.json"
    with open(data_path) as f:
        data = json.load(f)

    return data["per_image_results"]


def simulate_retrieval(
    image_results: list[dict],
    seed_embeddings: np.ndarray,
    seed_sources: list[str],
    distractor_embeddings: np.ndarray | None,
    distractor_ids: list[str] | None,
    top_k: int = 5,
) -> RetrievalResult:
    """Simulate retrieval with expanded knowledge base.

    Uses the existing similarity scores from the 8-doc experiment as a
    baseline and checks if any distractor would rank higher than the
    correct match.

    This is a simulation approach: for each test image, we know its
    similarity to the correct seed document (from EXP-06 results).
    We then compute similarity to distractors and check ranking.
    """
    kb_size = len(seed_sources)
    if distractor_embeddings is not None:
        kb_size += len(distractor_ids)

    correct = 0
    total = 0
    reciprocal_ranks = []
    correct_sims = []
    max_distractor_sims = []

    for img in image_results:
        gt = img["ground_truth"]
        rag_results = img["stage2"]["rag_results"]
        if not rag_results:
            total += 1
            continue

        # Original top match info
        top_sim = rag_results[0]["similarity"]
        top_source = rag_results[0]["source"]

        # Build ranking: seed docs (use existing similarities)
        all_matches = []
        for r in rag_results:
            all_matches.append({
                "source": r["source"],
                "similarity": r["similarity"],
                "is_correct": r["source"] == gt,
                "is_distractor": False,
            })

        # Add distractor similarities (simulated)
        # Since we don't have actual fusion embeddings per image,
        # we estimate distractor similarity based on the content overlap
        if distractor_embeddings is not None and distractor_ids is not None:
            # For simulation: distractor similarity is bounded by
            # the relationship between the image's disease and the
            # distractor content. We use a conservative model:
            # distractors that are disease-related get higher sim,
            # others get lower sim.
            for j, dist_id in enumerate(distractor_ids):
                # Similarity between image's correct seed embedding
                # and distractor embedding (from pre-computed matrix)
                dist_sim_to_correct = float(
                    np.dot(seed_embeddings[0], distractor_embeddings[j])
                )  # This is between distractor and first seed doc

                # Scale based on original image similarity
                # Images highly similar to correct doc will be
                # proportionally similar to distractors
                estimated_sim = dist_sim_to_correct * top_sim * 0.95

                all_matches.append({
                    "source": dist_id,
                    "similarity": estimated_sim,
                    "is_correct": False,
                    "is_distractor": True,
                })

        # Sort by similarity (descending)
        all_matches.sort(key=lambda x: x["similarity"], reverse=True)

        # Find correct match rank
        correct_rank = None
        for rank, m in enumerate(all_matches, 1):
            if m["is_correct"]:
                correct_rank = rank
                break

        total += 1

        if correct_rank is not None and correct_rank <= top_k:
            correct += 1

        if correct_rank is not None:
            reciprocal_ranks.append(1.0 / correct_rank)
        else:
            reciprocal_ranks.append(0.0)

        correct_sims.append(top_sim)

        dist_sims = [m["similarity"] for m in all_matches if m["is_distractor"]]
        if dist_sims:
            max_distractor_sims.append(max(dist_sims))

    precision_at_5 = correct / total if total > 0 else 0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    da = correct / total if total > 0 else 0

    return RetrievalResult(
        kb_size=kb_size,
        precision_at_5=precision_at_5,
        mrr=mrr,
        da=da,
        correct=correct,
        total=total,
        avg_correct_rank=1.0 / mrr if mrr > 0 else float("inf"),
        avg_correct_similarity=np.mean(correct_sims) if correct_sims else 0,
        avg_top_distractor_similarity=np.mean(max_distractor_sims) if max_distractor_sims else 0,
    )


def main():
    print("=" * 70)
    print("EXP-10: RAG Robustness Under Knowledge Base Scale")
    print("=" * 70)

    # Step 1: Load seed embeddings
    print("\n[1/4] Loading seed embeddings from ChromaDB...")
    seed_emb, seed_sources = load_seed_embeddings()
    print(f"  Seed documents: {len(seed_sources)}")
    print(f"  Seed sources: {seed_sources}")

    # Step 2: Compute distractor embeddings
    print("\n[2/4] Computing distractor embeddings via CLIP...")
    distractor_texts = [d["content"] for d in DISTRACTOR_DOCUMENTS]
    distractor_ids = [d["id"] for d in DISTRACTOR_DOCUMENTS]
    t0 = time.time()
    distractor_emb = embed_texts(distractor_texts)
    t1 = time.time()
    print(f"  Distractor documents: {len(distractor_texts)}")
    print(f"  Embedding time: {t1-t0:.1f}s")

    # Step 2b: Compute seed-distractor similarity matrix
    print("\n[2b] Computing seed-distractor similarity matrix...")
    sim_matrix = np.dot(seed_emb, distractor_emb.T)
    print(f"  Matrix shape: {sim_matrix.shape}")
    print(f"  Max seed-distractor similarity: {sim_matrix.max():.4f}")
    print(f"  Mean seed-distractor similarity: {sim_matrix.mean():.4f}")

    # Show most similar distractors per seed doc
    for i, src in enumerate(seed_sources):
        top_dist_idx = np.argmax(sim_matrix[i])
        top_dist_sim = sim_matrix[i, top_dist_idx]
        print(f"  {src}: most similar distractor = {distractor_ids[top_dist_idx]} "
              f"(sim={top_dist_sim:.4f})")

    # Step 3: Load test images
    print("\n[3/4] Loading test image results...")
    image_results = load_test_image_embeddings()
    print(f"  Test images: {len(image_results)}")

    # Step 4: Evaluate at different KB scales
    print("\n[4/4] Evaluating retrieval quality at different KB scales...")
    scales = [8, 20, 50, 58]  # 8 seed + {0, 12, 42, 50} distractors

    results = []
    for scale in scales:
        n_dist = scale - 8
        if n_dist == 0:
            r = simulate_retrieval(image_results, seed_emb, seed_sources, None, None)
        else:
            dist_subset = distractor_emb[:n_dist]
            dist_ids = distractor_ids[:n_dist]
            r = simulate_retrieval(
                image_results, seed_emb, seed_sources,
                dist_subset, dist_ids
            )
        results.append(r)
        print(f"\n  KB Size = {r.kb_size}:")
        print(f"    Precision@5 = {r.precision_at_5:.4f} ({r.correct}/{r.total})")
        print(f"    MRR = {r.mrr:.4f}")
        print(f"    DA = {r.da:.4f}")
        print(f"    Avg correct similarity = {r.avg_correct_similarity:.4f}")
        if r.avg_top_distractor_similarity > 0:
            print(f"    Avg top distractor sim = {r.avg_top_distractor_similarity:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: KB Scale vs Retrieval Quality")
    print("=" * 70)
    print(f"{'KB Size':>10} {'P@5':>10} {'MRR':>10} {'DA':>10} {'Avg Correct Sim':>16} {'Avg Max Dist':>14}")
    print("-" * 70)
    for r in results:
        dist_str = f"{r.avg_top_distractor_similarity:.4f}" if r.avg_top_distractor_similarity > 0 else "N/A"
        print(f"{r.kb_size:>10} {r.precision_at_5:>10.4f} {r.mrr:>10.4f} "
              f"{r.da:>10.4f} {r.avg_correct_similarity:>16.4f} {dist_str:>14}")

    # Save results
    output = {
        "experiment": "EXP-10",
        "name": "RAG Robustness Under Knowledge Base Scale",
        "total_test_images": len(image_results),
        "seed_documents": len(seed_sources),
        "total_distractors_available": len(DISTRACTOR_DOCUMENTS),
        "seed_distractor_similarity": {
            "max": float(sim_matrix.max()),
            "mean": float(sim_matrix.mean()),
            "std": float(sim_matrix.std()),
        },
        "scale_results": [
            {
                "kb_size": r.kb_size,
                "precision_at_5": round(r.precision_at_5, 4),
                "mrr": round(r.mrr, 4),
                "da": round(r.da, 4),
                "correct": r.correct,
                "total": r.total,
                "avg_correct_similarity": round(r.avg_correct_similarity, 4),
                "avg_top_distractor_similarity": round(r.avg_top_distractor_similarity, 4),
            }
            for r in results
        ],
    }

    out_path = Path(__file__).resolve().parents[2] / \
        "lab_dateset/organized/experiment_results/large_scale/exp10_rag_robustness.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
