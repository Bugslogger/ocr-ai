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
            "required": list(properties), "additionalProperties": False}


STRING = {"type": ["string", "null"]}
NUMBER = {"type": ["number", "null"]}
INTEGER = {"type": ["integer", "null"]}
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


def validate_invoice(value):
    from jsonschema import Draft202012Validator

    errors = sorted(Draft202012Validator(INVOICE_SCHEMA).iter_errors(value),
                    key=lambda e: str(list(e.absolute_path)))
    if errors:
        error = errors[0]
        location = ".".join(map(str, error.absolute_path)) or "root"
        raise ValueError(f"Invalid invoice JSON at {location}: {error.message}")
    return value
