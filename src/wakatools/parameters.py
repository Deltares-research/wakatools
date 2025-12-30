import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point

from wakatools.constants import SeismicVelocity


class SeismicCollection:
    # Vraag aan bas: Happy new year! oke ik dacht dus, dit is handig, want dan kunnen
    # we net als in borehole collection alle data netjes in 1 format
    # gebruiken en dan kan evt ook die to depth hier ook in ?
    # ik zet er even mee te stoeien of dit handig is ik denk van wel
    # maar weet het niet zeker dus dit is mn test beginnetje!
    # oh en mis smoet dit dan in base net als bij geost maar deze was ff leeg dus
    # ik heb m nu hier gezet want dan kan het er gelijk uit als het toch een
    # slecht idee is.
    # en nog een vraag, kunnen we samen kijken naar hoe ik hier een goede
    # test voor schrijf? :D

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_points_by_line(self, line_id):
        return self.df[self.df["ID"] == line_id]

    @staticmethod
    def _calculate_absolute_time(line1: LineString, line2: LineString) -> np.ndarray:
        """
        Calculate the absolute time difference between two seismic lines.

        """
        return np.array(
            [
                p.z - line1.interpolate(line1.project(p)).z
                for p in map(lambda c: Point(c), line2.coords)
            ]
        )

    def _time_to_depth(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert time to depth relative to bathymetry reflector
        for a single seismic line.

        """

        bathy_df = df[df["reflector"] == "bathy"]
        if bathy_df.empty:
            raise ValueError("No bathy reflector found")

        bathy_line = LineString(bathy_df[["x", "y", "time"]].values)

        depth = pd.Series(index=df.index, dtype=float)

        for ref in df["reflector"].unique():
            if ref == "bathy":
                continue

            ref_df = df[df["reflector"] == ref]
            ref_line = LineString(ref_df[["x", "y", "time"]].values)

            time = self._calculate_absolute_time(bathy_line, ref_line)

            depth.loc[ref_df.index] = time * (SeismicVelocity.SEDIMENT / 2.0)

        return depth

    def calc_depth(self) -> pd.Series:
        """
        Calculate depth for all data, grouped per seismic line (ID).

        """
        if self.df["ID"].nunique() == 1:
            depth = self._time_to_depth(self.df)
        else:
            depth = self.df.groupby("ID", group_keys=False).apply(self._time_to_depth)

        return depth.fillna(0.0)
