"""Invoice extraction contract shared by the prompt and validator."""
ORDER_FIELDS = (
    "company_name company_address invoice_number invoice_date purchase_order_number "
    "currency payment_terms inco_terms tax_id tracking_number shipper_company "
    "shipper_country receiver_company receiver_firstname receiver_lastname "
    "receiver_phone_number receiver_country receiver_city receiver_zipcode receiver_address1"
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
            "required": list(properties), "additionalProperties": False}


STRING = {"type": ["string", "null"]}
NUMBER = {"type": ["number", "null"]}
INVOICE_SCHEMA = object_schema({
    "invoices": {"type": "array", "minItems": 1, "items": object_schema({
        "order_level": object_schema({**{k: STRING for k in ORDER_FIELDS}, "total_amount": NUMBER}),
        "item_level": {"type": "array", "minItems": 1, "items": object_schema({
            **{k: STRING for k in ITEM_FIELDS}, **{k: NUMBER for k in NUMBER_FIELDS},
        })},
        "shipment": object_schema({k: STRING for k in ("invoice_number", "invoice_date", "currency")}),
    })},
})


def validate_invoice(value):
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(INVOICE_SCHEMA).iter_errors(value),
                    key=lambda e: str(list(e.absolute_path)))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "root"
        raise ValueError(f"Invalid invoice JSON at {location}: {error.message}")
    return value
