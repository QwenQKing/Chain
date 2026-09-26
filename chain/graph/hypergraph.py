from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from chain import config as cfg
from chain.graph.completion import load_graph_bundle_portable

logger = logging.getLogger(__name__)


class MMDTHypergraph:
    def __init__(
        self,
        kb_dir: Optional[str] = None,
        *,
        validate: bool = True,
        expected_profile_id: Optional[str] = None,
        expected_config: Any = None,
        require_publishable: bool = False,
        embedder: Any = None,
    ):
        self.kb_dir = str(kb_dir or cfg.HYPERGRAPH_DIR)
        self._expected_profile_id = expected_profile_id
        self._expected_config = expected_config
        self._require_publishable = require_publishable
        self._embedder = embedder
        self._engine = None
        self._validation: Optional[Dict[str, Any]] = None
        self._pending_reset = False
        if validate:
            self._validate_and_load()

    def _validate_and_load(self) -> None:
        from chain.graph.kg_builder import KGEngine




        validation = load_graph_bundle_portable(
            self.kb_dir,
            expected_config=self._expected_config,
            expected_profile_id=self._expected_profile_id,
            require_publishable=self._require_publishable,
        )
        self._validation = validation
        self._engine = KGEngine(
            self.kb_dir,
            resolved_config=self._expected_config,
            expected_profile_id=self._expected_profile_id,
            validate=True,
            embedder=self._embedder,
        )
        graph = self._engine._graph._graph
        graph.graph.update(
            {
                "validated_graph_bundle": True,
                "graph_bundle_sha256": validation["graph_bundle_sha256"],
                "profile_id": validation["profile_id"],
                "publishable": validation["publishable"],
                "out_of_paper_protocol": validation["out_of_paper_protocol"],
                "construction_config_sha256": validation["manifest"][
                    "construction_config_sha256"
                ],
                "scientific_config_sha256": validation["manifest"]["scientific_config_sha256"],
                "embedding_signature": validation["embedding_signatures"]["entity"],
            }
        )
        self._pending_reset = False

    @property
    def engine(self):
        if self._engine is None:
            self._validate_and_load()
        return self._engine

    @property
    def validation(self) -> Dict[str, Any]:
        if self._validation is None:
            self._validate_and_load()
        return dict(self._validation)

    @property
    def graph_bundle_sha256(self) -> str:
        return str(self.validation["graph_bundle_sha256"])

    @property
    def embedding_signature(self) -> Dict[str, Any]:
        return dict(self.validation["embedding_signatures"]["entity"])

    def reset(self) -> None:


        self._engine = None
        self._validation = None
        self._pending_reset = True
        logger.info("Hypergraph reset requested; existing target retained until atomic publish")

    def ingest_file(self, file_path: str, *, source_descriptor: Optional[str] = None) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge file not found: {file_path}")



        del source_descriptor
        return self.ingest(str(path))

    def ingest(self, path: str, source_date: str = "") -> Dict[str, Any]:
        if source_date:
            raise ValueError(
                "source_date override is no longer accepted; dates must come from each canonical record"
            )
        from chain.config import resolve_config
        from chain.graph.kg_builder import KGEngine

        config = self._expected_config or resolve_config("compat")
        writer = KGEngine(
            self.kb_dir,
            resolved_config=config,
            expected_profile_id=None,
            validate=False,
            embedder=self._embedder,
        )
        stats = writer.ingest(path)
        self._expected_config = config
        self._expected_profile_id = getattr(config, "profile_id", "compat")
        self._require_publishable = bool(getattr(config, "publishable", False))
        self._validate_and_load()
        logger.info("Ingested %s: %s", path, stats)
        return stats

    def build_replacement_bundle(
        self,
        source_path: str,
        *,
        source_descriptor: str,
        config: Any,
        force: bool = False,
    ) -> Dict[str, Any]:


        from chain.graph.kg_builder import build_graph_bundle

        if not isinstance(source_descriptor, str) or not source_descriptor.strip():
            raise ValueError("source_descriptor must be a non-empty stable identifier")
        if not isinstance(config, cfg.ResolvedConfig):
            raise TypeError("config must be a resolved configuration")
        allow_compat = config.profile != cfg.PAPER_PROFILE
        stats = build_graph_bundle(
            source_path,
            self.kb_dir,
            resolved_config=config,
            profile=config.profile,
            extractor=None,
            embedder=self._embedder,
            force=force,
            allow_compat_adapters=allow_compat,
            require_extraction_trace=True,
            source_descriptor=source_descriptor,
        )
        self._expected_config = config
        self._expected_profile_id = config.profile_id
        self._require_publishable = config.publishable
        self._engine = None
        self._validation = None
        self._validate_and_load()
        return {
            "stats": stats,
            "validation": self.validation,
            "graph_bundle_sha256": self.graph_bundle_sha256,
        }

    def graph_data(self) -> Dict[str, Any]:
        return self.engine.graph_data()

    def query(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:


        return self.retrieve(question, top_k=top_k)

    def retrieve(self, question: str, top_k: int = 10) -> List[Dict[str, Any]]:
        return self.engine.retrieve(question, top_k=top_k)

    def search_entity_vectors(
        self,
        query_vector: Sequence[float],
        candidate_ids: Optional[Set[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:


        hits = self.engine.search_entity_vectors(
            list(query_vector),
            top_k=top_k,
            candidate_node_ids=set(candidate_ids) if candidate_ids is not None else None,
        )
        return [
            {
                **hit,
                "entity_id": hit["node_id"],
            }
            for hit in hits
        ]

    def get_graph(self):


        return self.engine._graph._graph

    def stats(self) -> Dict[str, Any]:
        return self.engine.stats()

    def fork_subgraph(self, node_ids: List[str]):
        return self.get_graph().subgraph(node_ids).copy()


__all__ = ["MMDTHypergraph"]
