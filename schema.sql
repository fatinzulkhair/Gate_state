-- ============================================================
-- SCHEMA: Product Hierarchy (PMI Track & Trace style JSON)
-- Versi 2 — join key pakai PrimaryProductKey / natural key (bukan surrogate int id)
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- Master / reference tables (natural key sebagai PK)
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS customers (
    code    TEXT PRIMARY KEY,      -- Customer.Code, jadi join key
    name    TEXT,
    owner   TEXT
);

-- BrandFamily self-referencing (name dipakai sebagai key karena unik: IQOS, SNGLELUP, dst)
CREATE TABLE IF NOT EXISTS brand_families (
    name            TEXT PRIMARY KEY,
    description     TEXT,
    parent_name     TEXT,
    FOREIGN KEY (parent_name) REFERENCES brand_families(name)
);

CREATE TABLE IF NOT EXISTS product_definitions (
    definition_key      TEXT PRIMARY KEY,   -- ProductDefinitionKey, jadi join key
    description          TEXT,
    product_category     TEXT,
    brand_family_name    TEXT,
    FOREIGN KEY (brand_family_name) REFERENCES brand_families(name)
);

CREATE TABLE IF NOT EXISTS product_definition_properties (
    definition_key   TEXT NOT NULL,
    prop_name        TEXT NOT NULL,
    prop_value       TEXT,
    PRIMARY KEY (definition_key, prop_name),
    FOREIGN KEY (definition_key) REFERENCES product_definitions(definition_key)
);

-- ------------------------------------------------------------
-- Tabel per level produk (sesuai ProductType di JSON contoh).
-- Masing-masing pakai PrimaryProductKey sebagai PRIMARY KEY,
-- dan parent_product_key untuk join ke tabel parent-nya.
-- Kalau ada ProductType baru yang belum terdaftar di sini,
-- import_json_to_sqlite.py akan otomatis membuat tabel baru
-- dengan struktur yang sama (lihat fungsi ensure_product_table).
-- ------------------------------------------------------------

-- Level teratas: Shipping Case (tidak punya parent)
CREATE TABLE IF NOT EXISTS shipping_case (
    primary_product_key    TEXT PRIMARY KEY,
    parent_product_key     TEXT,           -- NULL untuk level teratas
    parent_table           TEXT,
    customer_code          TEXT,
    definition_key         TEXT,
    product_type_name      TEXT,
    quantity                REAL,
    FOREIGN KEY (customer_code) REFERENCES customers(code),
    FOREIGN KEY (definition_key) REFERENCES product_definitions(definition_key)
);

-- Level kedua: Bundle (child dari Shipping Case)
CREATE TABLE IF NOT EXISTS bundle (
    primary_product_key    TEXT PRIMARY KEY,
    parent_product_key     TEXT,
    parent_table           TEXT,
    customer_code          TEXT,
    definition_key         TEXT,
    product_type_name      TEXT,
    quantity                REAL,
    FOREIGN KEY (parent_product_key) REFERENCES shipping_case(primary_product_key),
    FOREIGN KEY (customer_code) REFERENCES customers(code),
    FOREIGN KEY (definition_key) REFERENCES product_definitions(definition_key)
);

-- Level ketiga: Pack (child dari Bundle)
CREATE TABLE IF NOT EXISTS pack (
    primary_product_key    TEXT PRIMARY KEY,
    parent_product_key     TEXT,
    parent_table           TEXT,
    customer_code          TEXT,
    definition_key         TEXT,
    product_type_name      TEXT,
    quantity                REAL,
    FOREIGN KEY (parent_product_key) REFERENCES bundle(primary_product_key),
    FOREIGN KEY (customer_code) REFERENCES customers(code),
    FOREIGN KEY (definition_key) REFERENCES product_definitions(definition_key)
);

-- ------------------------------------------------------------
-- Tabel anak generik (dipakai oleh SEMUA level produk),
-- di-join memakai primary_product_key milik produk terkait.
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS product_properties (
    primary_product_key    TEXT NOT NULL,
    prop_name               TEXT NOT NULL,
    prop_value               TEXT,
    PRIMARY KEY (primary_product_key, prop_name)
);

CREATE TABLE IF NOT EXISTS units (
    primary_product_key    TEXT NOT NULL,
    unit_type_name          TEXT,
    quantity                 REAL,
    base_unit                TEXT
);

CREATE TABLE IF NOT EXISTS alternate_product_keys (
    primary_product_key    TEXT NOT NULL,
    product_key              TEXT NOT NULL,
    product_key_type         INTEGER,
    product_key_mode         INTEGER
);

CREATE INDEX IF NOT EXISTS idx_bundle_parent ON bundle(parent_product_key);
CREATE INDEX IF NOT EXISTS idx_pack_parent ON pack(parent_product_key);
CREATE INDEX IF NOT EXISTS idx_props_key ON product_properties(primary_product_key);
CREATE INDEX IF NOT EXISTS idx_units_key ON units(primary_product_key);
CREATE INDEX IF NOT EXISTS idx_altkeys_key ON alternate_product_keys(primary_product_key);
