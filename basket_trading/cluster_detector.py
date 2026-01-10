"""
Copycat Cluster Detection.
Identifies and groups wallets that copy the same traders.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass

from api_clients.base import Trade

from . import BasketWallet, BasketConfig


@dataclass
class Cluster:
    """A group of wallets with similar trading patterns."""
    cluster_id: str
    wallet_addresses: List[str]
    similarity_score: float
    representative_address: str


class ClusterDetector:
    """
    Detects copycat clusters among wallet baskets.

    Two wallets are in the same cluster if:
    - They trade the same markets (>85% similarity)
    - They enter positions at similar times
    - They have similar position sizes

    Only the representative wallet from each cluster
    should participate in basket signals.
    """

    def __init__(self, config: BasketConfig):
        """Initialize cluster detector."""
        self.config = config
        self.similarity_threshold = config.cluster_similarity_threshold
        self.max_cluster_size = config.max_cluster_size

    def detect_clusters(
        self,
        basket_wallets: List[BasketWallet],
        wallet_trades: Dict[str, List[Trade]],
    ) -> List[Cluster]:
        """Detect copycat clusters among basket wallets."""
        if len(basket_wallets) < 2:
            return []

        logger.info(f"Detecting clusters among {len(basket_wallets)} wallets")

        addresses = [w.wallet_address for w in basket_wallets]

        similarity_matrix = self._calculate_similarity_matrix(
            addresses,
            wallet_trades,
        )

        clusters = self._build_clusters_from_similarity(
            addresses,
            similarity_matrix,
        )

        logger.info(f"Found {len(clusters)} clusters")

        return clusters

    def assign_clusters_to_basket(
        self,
        basket_wallets: List[BasketWallet],
        clusters: List[Cluster],
    ) -> List[BasketWallet]:
        """Assign cluster IDs to basket wallets."""
        cluster_map: Dict[str, Cluster] = {}
        for cluster in clusters:
            cluster_map[cluster.representative_address] = cluster
            for addr in cluster.wallet_addresses:
                cluster_map[addr] = cluster

        updated_wallets = []
        for wallet in basket_wallets:
            cluster = cluster_map.get(wallet.wallet_address)
            if cluster:
                wallet.cluster_id = cluster.cluster_id
                wallet.is_cluster_representative = (
                    wallet.wallet_address == cluster.representative_address
                )

            updated_wallets.append(wallet)

        cluster_count = len(set(w.cluster_id for w in updated_wallets if w.cluster_id))
        logger.info(f"Assigned {cluster_count} clusters to basket wallets")

        return updated_wallets

    def _calculate_similarity_matrix(
        self,
        addresses: List[str],
        wallet_trades: Dict[str, List[Trade]],
    ) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise similarity between all wallets."""
        similarity_matrix: Dict[Tuple[str, str], float] = {}

        for i, addr1 in enumerate(addresses):
            for addr2 in addresses[i + 1:]:
                similarity = self._calculate_wallet_similarity(
                    addr1,
                    addr2,
                    wallet_trades.get(addr1, []),
                    wallet_trades.get(addr2, []),
                )
                similarity_matrix[(addr1, addr2)] = similarity
                similarity_matrix[(addr2, addr1)] = similarity

        return similarity_matrix

    def _calculate_wallet_similarity(
        self,
        addr1: str,
        addr2: str,
        trades1: List[Trade],
        trades2: List[Trade],
    ) -> float:
        """Calculate similarity score between two wallets."""
        if not trades1 or not trades2:
            return 0.0

        markets1 = set(t.market_id for t in trades1)
        markets2 = set(t.market_id for t in trades2)

        intersection = markets1 & markets2
        union = markets1 | markets2

        if not union:
            return 0.0

        market_similarity = len(intersection) / len(union)

        timing_similarity = self._calculate_timing_similarity(trades1, trades2)
        size_similarity = self._calculate_size_similarity(trades1, trades2)

        overall_similarity = (
            0.6 * market_similarity +
            0.2 * timing_similarity +
            0.2 * size_similarity
        )

        return overall_similarity

    def _calculate_timing_similarity(
        self,
        trades1: List[Trade],
        trades2: List[Trade],
    ) -> float:
        """Calculate timing similarity between wallets."""
        if not trades1 or not trades2:
            return 0.0

        trades1_by_market = defaultdict(list)
        trades2_by_market = defaultdict(list)

        for t in trades1:
            trades1_by_market[t.market_id].append(t.timestamp)

        for t in trades2:
            trades2_by_market[t.market_id].append(t.timestamp)

        common_markets = set(trades1_by_market.keys()) & set(trades2_by_market.keys())

        if not common_markets:
            return 0.0

        similar_entries = 0
        total_entries = 0

        for market_id in common_markets:
            times1 = trades1_by_market[market_id]
            times2 = trades2_by_market[market_id]

            for t1 in times1:
                for t2 in times2:
                    time_diff = abs((t1 - t2).total_seconds())
                    if time_diff < 300:
                        similar_entries += 1
                    total_entries += 1

        return similar_entries / total_entries if total_entries > 0 else 0.0

    def _calculate_size_similarity(
        self,
        trades1: List[Trade],
        trades2: List[Trade],
    ) -> float:
        """Calculate position size similarity between wallets."""
        if not trades1 or not trades2:
            return 0.0

        sizes1 = [t.quantity for t in trades1]
        sizes2 = [t.quantity for t in trades2]

        avg_size1 = sum(sizes1) / len(sizes1)
        avg_size2 = sum(sizes2) / len(sizes2)

        if avg_size1 == 0 or avg_size2 == 0:
            return 0.0

        size_ratio = min(avg_size1, avg_size2) / max(avg_size1, avg_size2)

        return size_ratio

    def _build_clusters_from_similarity(
        self,
        addresses: List[str],
        similarity_matrix: Dict[Tuple[str, str], float],
    ) -> List[Cluster]:
        """Build clusters from similarity matrix using greedy approach."""
        assigned: Set[str] = set()
        clusters: List[Cluster] = []
        cluster_id_counter = 0

        for addr in addresses:
            if addr in assigned:
                continue

            similar_wallets = [addr]
            for other_addr in addresses:
                if other_addr != addr and other_addr not in assigned:
                    similarity = similarity_matrix.get((addr, other_addr), 0.0)
                    if similarity >= self.similarity_threshold:
                        similar_wallets.append(other_addr)

            if len(similar_wallets) > 1:
                cluster_id = f"cluster_{cluster_id_counter}"
                cluster_id_counter += 1

                representative = self._select_representative(
                    similar_wallets,
                    similarity_matrix,
                )

                cluster = Cluster(
                    cluster_id=cluster_id,
                    wallet_addresses=similar_wallets,
                    similarity_score=sum(
                        similarity_matrix.get((representative, w), 0.0)
                        for w in similar_wallets
                    ) / len(similar_wallets),
                    representative_address=representative,
                )

                clusters.append(cluster)
                assigned.update(similar_wallets)
            else:
                assigned.add(addr)

        return clusters

    def _select_representative(
        self,
        addresses: List[str],
        similarity_matrix: Dict[Tuple[str, str], float],
    ) -> str:
        """Select the best wallet to represent the cluster."""
        best_addr = addresses[0]
        best_score = 0.0

        for addr in addresses:
            avg_similarity = sum(
                similarity_matrix.get((addr, other), 0.0)
                for other in addresses
                if other != addr
            ) / max(1, len(addresses) - 1)

            if avg_similarity > best_score:
                best_score = avg_similarity
                best_addr = addr

        return best_addr


logger = __import__('logging').getLogger(__name__)
