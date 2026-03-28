// ============================================================
// AOI Greater Guayaquil + Manglares SERVIR-Amazonia
// GEE Code Editor: https://code.earthengine.google.com/
// ============================================================

// ======================== CANTONES ========================

var gaul2 = ee.FeatureCollection('FAO/GAUL/2015/level2');

var cantones = gaul2.filter(ee.Filter.and(
  ee.Filter.eq('ADM0_NAME', 'Ecuador'),
  ee.Filter.inList('ADM2_NAME', ['Guayaquil', 'Duran', 'Daule', 'Samborondon'])
));

print('Cantones encontrados:', cantones.size());
print('Nombres:', cantones.aggregate_array('ADM2_NAME'));

// ======================== AOI ========================

var estuaryBbox = ee.Geometry.BBox(-80.35, -2.85, -79.55, -1.85);
var cantonesClipped = cantones.geometry().intersection(estuaryBbox, 1);
var aoiBuffered = cantonesClipped.buffer(3000);
var aoiFinal = aoiBuffered.intersection(estuaryBbox, 1);
var aoiFc = ee.FeatureCollection([ee.Feature(aoiFinal)]);

// ======================== MANGLARES SERVIR ========================

var man2018 = ee.FeatureCollection('projects/gee-space-hack/assets/SpaceHack/data/MAN_2018');
var man2020 = ee.FeatureCollection('projects/gee-space-hack/assets/SpaceHack/data/MAN_2020');
var man2022 = ee.FeatureCollection('projects/gee-space-hack/assets/SpaceHack/data/MAN_2022');
var cambios = ee.FeatureCollection('projects/gee-space-hack/assets/SpaceHack/data/Cambios_18_22');

print('--- Estadisticas ---');
print('Poligonos manglar 2018:', man2018.size());
print('Poligonos manglar 2020:', man2020.size());
print('Poligonos manglar 2022:', man2022.size());
print('Poligonos cambios 2018-2022:', cambios.size());

// Inspeccionar estructura de la capa de cambios
print('Propiedades cambios (primer feature):', cambios.first());

// ======================== S2 FONDO ========================

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoiFinal)
  .filterDate('2022-06-01', '2022-09-30')
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

// ======================== VISUALIZACION ========================

Map.centerObject(aoiFc, 10);

// Fondo
Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'S2 RGB 2022', false);
Map.addLayer(s2, {bands: ['B8', 'B4', 'B3'], min: 0, max: 4000}, 'S2 Falso Color', false);

// Limites
Map.addLayer(cantones, {color: '3388ff'}, 'Cantones', false);
Map.addLayer(aoiFc, {color: 'ff0000'}, 'AOI final');

// Manglares por ano
Map.addLayer(man2018, {color: 'ff8800'}, 'Manglar 2018');
Map.addLayer(man2020, {color: '00cc66'}, 'Manglar 2020');
Map.addLayer(man2022, {color: '0088ff'}, 'Manglar 2022');

// Cambios
Map.addLayer(cambios, {color: 'ff00ff'}, 'Cambios 2018-2022');

// ======================== EXPORTAR AOI ========================

Export.table.toAsset({
  collection: aoiFc,
  description: 'AOI_GreaterGuayaquil_v2',
  assetId: 'projects/gee-space-hack/assets/SpaceHack/AOI_GreaterGuayaquil_v2'
});
