"""
Unified compression configuration for Zarr v2 and v3.

Handles the differences between:
- Zarr v2: Uses numcodecs compressor objects
- Zarr v3: Uses codec pipeline (list of codec configurations)
"""


class Codecs:
    """
    Unified compression configuration for Zarr v2 and v3.
    
    Handles the differences between:
    - Zarr v2: Uses numcodecs compressor objects
    - Zarr v3: Uses codec pipeline (list of codec configurations)
    
    Examples:
        # Blosc compression with LZ4
        codecs = Codecs('blosc', clevel=5, cname='lz4')
        
        # ZSTD compression
        codecs = Codecs('zstd', clevel=3)
        
        # No compression
        codecs = Codecs(None)
    """
    
    def __init__(self, compressor='blosc', clevel=5, cname='lz4', shuffle=1,
                 typesize=None):
        """
        Initialize compression configuration.

        Args:
            compressor: Compression algorithm ('blosc', 'zstd', 'gzip', 'lz4', 'bz2', None)
            clevel: Compression level (1-9, higher = more compression but slower)
            cname: Blosc compressor name ('lz4', 'zstd', 'zlib', 'snappy', 'blosclz')
            shuffle: Blosc shuffle mode (0=no shuffle, 1=byte shuffle, 2=bit shuffle)
            typesize: Blosc element width IN BYTES, for the (bit)shuffle stage. ``None``
                (default) derives it from the array dtype at write time - which is what
                you want; pass an explicit value only to override.

                This matters, and only for Zarr v3. Both shuffle modes permute *within*
                elements (byte shuffle groups the Nth byte of every element; bitshuffle
                does the same per bit), so the element width IS the transform. Zarr v2
                does not store it - blosc infers it from the buffer's itemsize at encode
                time - but v3 serializes ``typesize`` into the array metadata, so it must
                be set explicitly or it defaults to 1 and the shuffle degenerates to a
                byte-wise permutation of the raw stream.

                Measured on int32 labels, zstd-5 bitshuffle, one 64^3 chunk:
                typesize=1 -> 25.6 KiB, typesize=4 -> 20.8 KiB (and faster to encode).
        """
        self.compressor = compressor
        self.clevel = clevel
        self.cname = cname
        self.shuffle = shuffle
        self.typesize = typesize

    @staticmethod
    def _typesize_for(dtype):
        """Element width in bytes for `dtype`, clamped to what blosc accepts (1..255)."""
        import numpy as np
        try:
            size = int(np.dtype(dtype).itemsize)
        except Exception:
            return 1
        return size if 1 <= size <= 255 else 1
    
    def to_v2_config(self):
        """
        Generate compressor configuration dict for Zarr v2 (TensorStore-compatible).
        
        Returns:
            Dictionary with compressor configuration or None
        """
        if self.compressor == 'blosc':
            return {
                'id': 'blosc',
                'cname': self.cname,
                'clevel': self.clevel,
                'shuffle': self.shuffle
            }
        elif self.compressor == 'zstd':
            return {
                'id': 'zstd',
                'level': self.clevel
            }
        elif self.compressor == 'gzip':
            return {
                'id': 'gzip',
                'level': self.clevel
            }
        elif self.compressor == 'lz4':
            return {
                'id': 'lz4'
            }
        elif self.compressor == 'bz2':
            return {
                'id': 'bz2',
                'level': self.clevel
            }
        elif self.compressor is None or self.compressor == 'none':
            return None
        else:
            raise ValueError(f"Unsupported compressor: {self.compressor}")
    
    def to_numcodecs(self):
        """
        Generate numcodecs compressor object for Zarr v2.
        
        Returns:
            numcodecs compressor object or None
        """
        if self.compressor == 'blosc':
            import numcodecs
            return numcodecs.Blosc(cname=self.cname, clevel=self.clevel, shuffle=self.shuffle)
        elif self.compressor == 'zstd':
            import numcodecs
            return numcodecs.Zstd(level=self.clevel)
        elif self.compressor == 'gzip':
            import numcodecs
            return numcodecs.GZip(level=self.clevel)
        elif self.compressor == 'lz4':
            import numcodecs
            return numcodecs.LZ4()
        elif self.compressor == 'bz2':
            import numcodecs
            return numcodecs.BZ2(level=self.clevel)
        elif self.compressor is None or self.compressor == 'none':
            return None
        else:
            raise ValueError(f"Unsupported compressor: {self.compressor}")
    
    def to_v3_config(self, dtype=None):
        """
        Generate codec pipeline for Zarr v3 using proper zarr.codecs API.

        Args:
            dtype: Array dtype, used to derive the blosc ``typesize`` when this Codecs was
                built without an explicit one. Zarr v3 SERIALIZES typesize into the array
                metadata (v2 does not - blosc infers it there), so leaving it unset writes
                ``typesize: 1`` and the (bit)shuffle silently degrades to a byte-wise
                permutation. Always pass the dtype you are writing.

        Returns:
            List of codec configurations (from .to_dict() calls)
        """
        from zarr import codecs

        codecs_list = [
            codecs.BytesCodec(endian=codecs.Endian.little).to_dict()
        ]

        if self.compressor == 'blosc':
            # Use BloscCodec with proper BloscShuffle enum
            shuffle_map = {0: codecs.BloscShuffle.noshuffle,
                          1: codecs.BloscShuffle.shuffle,
                          2: codecs.BloscShuffle.bitshuffle}
            shuffle_enum = shuffle_map.get(self.shuffle, codecs.BloscShuffle.shuffle)

            typesize = self.typesize
            if typesize is None:
                typesize = self._typesize_for(dtype) if dtype is not None else 1

            blosc_codec = codecs.BloscCodec(
                cname=self.cname,
                clevel=self.clevel,
                shuffle=shuffle_enum,
                typesize=typesize,
            )
            codecs_list.append(blosc_codec.to_dict())
            
        elif self.compressor == 'zstd':
            zstd_codec = codecs.ZstdCodec(level=self.clevel)
            codecs_list.append(zstd_codec.to_dict())
            
        elif self.compressor == 'gzip':
            gzip_codec = codecs.GzipCodec(level=self.clevel)
            codecs_list.append(gzip_codec.to_dict())
            
        elif self.compressor == 'lz4':
            # LZ4 is not directly available in zarr.codecs v3, would need blosc
            raise ValueError("LZ4 is not available in Zarr v3. Use Blosc with cname='lz4' instead.")
            
        elif self.compressor == 'bz2':
            # BZ2 is not directly available in zarr.codecs v3
            raise ValueError("BZ2 is not available in Zarr v3. Use Zstd or Gzip instead.")
            
        elif self.compressor is None or self.compressor == 'none':
            pass  # No compression codec, only bytes codec
            
        else:
            raise ValueError(f"Unsupported compressor: {self.compressor}")
        
        return codecs_list
    
    @classmethod
    def from_numcodecs(cls, compressor):
        """
        Create Codecs from a numcodecs compressor object.
        
        Args:
            compressor: numcodecs compressor object
            
        Returns:
            Codecs instance
        """
        if compressor is None:
            return cls(compressor=None)
        
        compressor_type = type(compressor).__name__.lower()
        
        if 'blosc' in compressor_type:
            return cls(
                compressor='blosc',
                clevel=getattr(compressor, 'clevel', 5),
                cname=getattr(compressor, 'cname', 'lz4'),
                shuffle=getattr(compressor, 'shuffle', 1)
            )
        elif 'zstd' in compressor_type:
            return cls(
                compressor='zstd',
                clevel=getattr(compressor, 'level', 5)
            )
        elif 'gzip' in compressor_type:
            return cls(
                compressor='gzip',
                clevel=getattr(compressor, 'level', 5)
            )
        elif 'lz4' in compressor_type:
            return cls(compressor='lz4')
        elif 'bz2' in compressor_type:
            return cls(
                compressor='bz2',
                clevel=getattr(compressor, 'level', 5)
            )
        else:
            raise ValueError(f"Unsupported numcodecs compressor: {type(compressor)}")
