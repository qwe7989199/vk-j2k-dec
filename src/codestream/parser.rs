struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    codestream: Codestream,
}

impl<'a> Parser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            codestream: Codestream {
                byte_len: 0,
                siz: None,
                cod: None,
                qcd: None,
                comments: Vec::new(),
                tile_parts: Vec::new(),
                markers: Vec::new(),
            },
        }
    }

    fn parse(mut self) -> Result<Codestream, ParseError> {
        let first = self.read_marker()?;
        if first != Marker::Soc {
            return Err(ParseError::new(0, "codestream must start with SOC"));
        }
        self.codestream.markers.push(MarkerRecord {
            offset: 0,
            marker: first,
        });

        while self.pos < self.data.len() {
            let marker_offset = self.pos;
            let marker = self.read_marker()?;
            self.codestream.markers.push(MarkerRecord {
                offset: marker_offset,
                marker,
            });

            match marker {
                Marker::Siz => {
                    let segment = self.read_segment(marker)?;
                    self.codestream.siz = Some(parse_siz(marker_offset, segment)?);
                }
                Marker::Cod => {
                    let segment = self.read_segment(marker)?;
                    self.codestream.cod = Some(parse_cod(marker_offset, segment)?);
                }
                Marker::Qcd => {
                    let segment = self.read_segment(marker)?;
                    self.codestream.qcd = Some(parse_qcd(marker_offset, segment)?);
                }
                Marker::Sot => {
                    let segment = self.read_segment(marker)?;
                    let mut sot = parse_sot(marker_offset, segment)?;
                    let tile_part_end = self.tile_part_end(sot.tile_part_length, marker_offset)?;
                    let (data_start, data_end) = self.parse_tile_part_until_sod(tile_part_end)?;
                    sot.data_start = data_start;
                    sot.data_end = data_end;
                    self.codestream.tile_parts.push(sot);
                }
                Marker::Sod => {
                    self.skip_entropy_data(None)?;
                }
                Marker::Eoc => {
                    self.codestream.byte_len = self.pos;
                    return Ok(self.codestream);
                }
                Marker::Coc
                | Marker::Qcc
                | Marker::Rgn
                | Marker::Pod
                | Marker::Ppm
                | Marker::Tlm
                | Marker::Plm => {
                    let _ = self.read_segment(marker)?;
                }
                Marker::Com => {
                    let segment = self.read_segment(marker)?;
                    self.codestream
                        .comments
                        .push(parse_com(marker_offset, segment)?);
                }
                Marker::Soc => {
                    return Err(ParseError::new(marker_offset, "unexpected SOC marker"));
                }
                Marker::Unknown(_) => {
                    return Err(ParseError::new(
                        marker_offset,
                        format!("unknown marker without segment handling: {marker}"),
                    ));
                }
            }
        }

        Err(ParseError::new(self.data.len(), "missing EOC marker"))
    }

    fn parse_tile_part_until_sod(
        &mut self,
        tile_part_end: Option<usize>,
    ) -> Result<(usize, usize), ParseError> {
        loop {
            if let Some(end) = tile_part_end
                && self.pos >= end
            {
                return Err(ParseError::new(self.pos, "tile-part ended before SOD"));
            }

            let marker_offset = self.pos;
            let marker = self.read_marker()?;
            self.codestream.markers.push(MarkerRecord {
                offset: marker_offset,
                marker,
            });

            match marker {
                Marker::Sod => {
                    let data_start = self.pos;
                    self.skip_entropy_data(tile_part_end)?;
                    let data_end = self.pos;
                    return Ok((data_start, data_end));
                }
                Marker::Qcd => {
                    let segment = self.read_segment(marker)?;
                    self.codestream.qcd = Some(parse_qcd(marker_offset, segment)?);
                }
                marker if marker.has_segment() => {
                    let _ = self.read_segment(marker)?;
                }
                _ => {
                    return Err(ParseError::new(
                        marker_offset,
                        format!("unexpected marker in tile-part header: {marker}"),
                    ));
                }
            }
        }
    }

    fn read_marker(&mut self) -> Result<Marker, ParseError> {
        if self.pos + 2 > self.data.len() {
            return Err(ParseError::new(self.pos, "expected marker"));
        }

        if self.data[self.pos] != 0xff {
            return Err(ParseError::new(self.pos, "expected marker prefix 0xff"));
        }

        let code = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(Marker::from_code(code))
    }

    fn read_segment(&mut self, marker: Marker) -> Result<&'a [u8], ParseError> {
        if self.pos + 2 > self.data.len() {
            return Err(ParseError::new(
                self.pos,
                format!("{marker} marker is missing segment length"),
            ));
        }

        let length_offset = self.pos;
        let length = read_u16(self.data, &mut self.pos, length_offset)? as usize;
        if length < 2 {
            return Err(ParseError::new(
                length_offset,
                "segment length must include L marker bytes",
            ));
        }

        let payload_len = length - 2;
        if self.pos + payload_len > self.data.len() {
            return Err(ParseError::new(
                self.pos,
                format!("{marker} segment extends past end of input"),
            ));
        }

        let segment = &self.data[self.pos..self.pos + payload_len];
        self.pos += payload_len;
        Ok(segment)
    }

    fn skip_entropy_data(&mut self, tile_part_end: Option<usize>) -> Result<(), ParseError> {
        if let Some(end) = tile_part_end {
            if end > self.data.len() {
                return Err(ParseError::new(
                    end,
                    "tile-part length extends past end of input",
                ));
            }
            self.pos = end;
            return Ok(());
        }

        while self.pos + 1 < self.data.len() {
            if self.data[self.pos] == 0xff {
                let next = self.data[self.pos + 1];
                if next != 0x00 && next != 0xff {
                    return Ok(());
                }
            }
            self.pos += 1;
        }

        Err(ParseError::new(self.pos, "missing marker after SOD"))
    }

    fn tile_part_end(
        &self,
        tile_part_length: u32,
        marker_offset: usize,
    ) -> Result<Option<usize>, ParseError> {
        if tile_part_length == 0 {
            return Ok(None);
        }

        let end = marker_offset
            .checked_add(tile_part_length as usize)
            .ok_or_else(|| ParseError::new(marker_offset, "tile-part length overflow"))?;
        if end > self.data.len() {
            return Err(ParseError::new(
                marker_offset,
                "tile-part length extends past end of input",
            ));
        }
        Ok(Some(end))
    }
}
