"""Invoice extraction contract shared by the prompt and validator."""
ORDER_FIELDS = (
    "company_name company_address invoice_number invoice_date purchase_order_number "
    "currency payment_terms inco_terms tax_id tracking_number shipper_company "
    "shipper_email shipper_phone_number shipper_country shipper_state shipper_city "
    "shipper_zipcode shipper_address1 shipper_address2 receiver_company receiver_firstname "
    "receiver_lastname receiver_phone_number receiver_country receiver_state receiver_city "
    "receiver_zipcode receiver_address1 receiver_address2 receiver_email gross_weight "
    "net_weight package_count total_quantity total_amount shipping_and_handling "
    "total_item_value dimensions_cm origin_phone_code destination_phone_code "
    "origin_country destination_country"
).split()
ORDER_NUMBER_FIELDS = (
    "gross_weight net_weight package_count total_quantity total_amount "
    "shipping_and_handling total_item_value"
).split()
ITEM_FIELDS = (
    "sku_code manufacturing_country_hts_code destination_country_hts_code hs_code "
    "product_description simplified_description country_of_manufacture"
).split()
NUMBER_FIELDS = (
    "gross_value number_of_units_for_this_sku unit_value_for_this_sku extended_value_for_this_sku"
).split()


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": [key for key, schema in properties.items() if "null" not in schema["type"]],
            "additionalProperties": False}


STRING = {"type": ["string", "null"]}
NUMBER = {"type": ["number", "null"]}
INTEGER = {"type": ["integer", "null"]}
COUNTRY_DESCRIPTIONS = {
    "origin_country": "Shipment departure/export country, not the product's country of origin or manufacture.",
    "shipper_country": "Country in the shipper/exporter's address; not proof of product origin.",
    "destination_country": "Actual goods delivery/Ship To country, separate from product origin and export country.",
    "receiver_country": "Country in the receiver's address.",
    "country_of_manufacture": (
        "Product country of origin/manufacture from explicit Country of Origin, COO, Made in, or Manufactured in evidence. "
        "Item-specific evidence takes precedence over a declaration covering all goods. Never substitute the export/shipper country; "
        "null when product-origin evidence is absent."
    ),
}
INVOICE_SCHEMA = object_schema({
    "invoices": {"type": "array", "minItems": 1, "items": object_schema({
        "order_level": object_schema({
            k: INTEGER if k == "package_count" else NUMBER if k in ORDER_NUMBER_FIELDS else STRING
            for k in ORDER_FIELDS
        }),
        "item_level": {"type": "array", "minItems": 1, "items": object_schema({
            **{k: STRING for k in ITEM_FIELDS}, **{k: NUMBER for k in NUMBER_FIELDS},
        })},
        "shipment": object_schema({k: STRING for k in ("invoice_number", "invoice_date", "currency")}),
    })},
})

# Describe existing fields without changing the API's required keys or types.
_invoice_properties = INVOICE_SCHEMA["properties"]["invoices"]["items"]["properties"]
for _group in (_invoice_properties["order_level"]["properties"],
               _invoice_properties["item_level"]["items"]["properties"]):
    for _field, _description in COUNTRY_DESCRIPTIONS.items():
        if _field in _group:
            _group[_field] = {**_group[_field], "description": _description}


def _fill_missing_fields(value, schema):
    """Materialize absent nullable fields without coercing supplied values."""
    if schema.get("type") == "object" and isinstance(value, dict):
        result = dict(value)
        for key, field_schema in schema["properties"].items():
            if key not in result:
                if "null" in field_schema["type"]:
                    result[key] = None
                elif field_schema["type"] == "object":
                    result[key] = {}
            if key in result:
                result[key] = _fill_missing_fields(result[key], field_schema)
        return result
    if schema.get("type") == "array" and isinstance(value, list):
        return [_fill_missing_fields(item, schema["items"]) for item in value]
    return value


def validate_invoice(value):
    from jsonschema import Draft202012Validator

    value = _fill_missing_fields(value, INVOICE_SCHEMA)
    errors = sorted(Draft202012Validator(INVOICE_SCHEMA).iter_errors(value),
                    key=lambda e: str(list(e.absolute_path)))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "root"
        raise ValueError(f"Invalid invoice JSON at {location}: {error.message}")
    return value
