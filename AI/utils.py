import re
import platform
import torch
import tempfile
import fitz  # PyMuPDF
import sys
from io import StringIO
import os
from huggingface_hub import login
import json
import requests
from IPython.display import display, JSON

def pdf_to_images(pdf_path, dpi=300):
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i+1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    print("pdf_to_images ready.")
    return paths


def check_GPU():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {platform.python_version()}")
    

def extract_doc(doc_path, output_path,model):
    PDF_PATH = os.environ.get("PDF_PATH", doc_path)
    OCR_OUTPUT_DIR  = ''
    if not output_path: 
        OCR_OUTPUT_DIR = "output/pdf"
    else:
        OCR_OUTPUT_DIR = output_path
    os.makedirs(OCR_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(PDF_PATH):
        print(f"WARNING: {PDF_PATH} not found. Upload the invoice and re-run this cell.")
        image_paths = []
    else:
        image_paths = pdf_to_images(PDF_PATH, dpi=350)
        print(f"Rendered {len(image_paths)} page image(s).")

    full_raw_ocr = ""

    if image_paths:
        print(f"Running OCR on {len(image_paths)} pages simultaneously...")
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = mystdout = StringIO()
        sys.stderr = StringIO()
        try:
            model.ocr_model.infer_multi(
                model.ocr_tokenizer,
                prompt="<image>Document parsing.",
                image_files=image_paths,
                output_path=OCR_OUTPUT_DIR,
                image_size=1024,
                max_length=15000,
                no_repeat_ngram_size=35,
                ngram_window=1024,
                save_results=True,
            )
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

        captured_output = mystdout.getvalue()

        # The model sometimes creates an empty or useless result.md. We'll use captured_output directly.
        result_md = os.path.join(OCR_OUTPUT_DIR, "result.md")
        if os.path.exists(result_md):
            os.remove(result_md) # Clean up to avoid confusion

        full_raw_ocr = captured_output

    detailed_path = os.path.join(OCR_OUTPUT_DIR, "detailed_invoice_output.txt")
    with open(detailed_path, "w", encoding="utf-8") as f:
        f.write(full_raw_ocr)
    print(f"Raw OCR saved: {detailed_path} ({len(full_raw_ocr)} chars)")
    

def clean_extracted_text(output_path):
    if not output_path: 
        OCR_OUTPUT_DIR = "output/pdf" 
    else: 
        OCR_OUTPUT_DIR = output_path
        
    detailed_output_file = os.path.join(OCR_OUTPUT_DIR, "detailed_invoice_output.txt")
    plain_text_output_file = os.path.join(OCR_OUTPUT_DIR, "plain_invoice_text.txt")

    with open(detailed_output_file, "r", encoding="utf-8") as f:
        content = f.read()

    cleaned_text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", content, flags=re.DOTALL)
    cleaned_text = re.sub(r"<[^>]*>", "", cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text).strip()

    with open(plain_text_output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Cleaned text saved: {plain_text_output_file} ({len(cleaned_text)} chars)")
    print("--- preview ---")
    print(cleaned_text[:1500] if cleaned_text else "[empty]")
    
def login_huggingface(HF_TOKEN):
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            hf_token = HF_TOKEN
        except Exception:
            hf_token = None

    if hf_token:
        login(token=hf_token)
        print("Logged in to Hugging Face.")
    else:
        print("No HF_TOKEN found. Public models may still load; gated models will fail.")
    
def format_json(Qtokenizer,Qmodel,system_prompt):
    def qwen_generate(messages, max_new_tokens=8000, temperature=0.0):
        prompt_text = Qtokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = Qtokenizer([prompt_text], return_tensors="pt").to(Qmodel.device)
        with torch.no_grad():
            out = Qmodel.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=Qtokenizer.eos_token_id,
                pad_token_id=Qtokenizer.eos_token_id,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        return Qtokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()


    def parse_json_maybe(text):
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))


    text_to_format = cleaned_text if "cleaned_text" in globals() and cleaned_text else ""
    if not text_to_format:
        raise RuntimeError("cleaned_text is empty. Run the OCR cells first.")

    print("Generating invoice JSON (no tariff classification)...")
    raw_json = qwen_generate(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"OCR Text:\n\n{text_to_format}"},
        ],
        max_new_tokens=40000,
        temperature=0.0,
    )
    parsed_json = parse_json_maybe(raw_json)

    # Hard-clear any leaked tariff guesses from extraction
    for inv in parsed_json.get("invoices", []):
        for item in inv.get("item_level", []):
            for field in (
                "hs_code",
                "manufacturing_country_hts_code",
                "destination_country_hts_code",
                "classification_reason",
            ):
                val = item.get(field)
                if isinstance(val, str) and re.match(r"^[A-Za-z][A-Za-z\s]*$", val.strip()):
                    item[field] = None
            # extraction step must not classify
            item["hs_code"] = None
            item["manufacturing_country_hts_code"] = None
            item["destination_country_hts_code"] = None
            item["classification_reason"] = None

    os.makedirs("/content/output", exist_ok=True)
    extract_path = "/content/output/extracted_invoice.json"
    with open(extract_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=2)
    print(f"Saved {extract_path}")
    print(json.dumps(parsed_json, indent=2)[:3000])
    
def fetch_hts():
    # Ensure weaviate_client is available from previous cells
    if 'weaviate_client' not in globals() or not weaviate_client:
        raise RuntimeError("Weaviate client not connected. Please run the Weaviate connection cells first.")

    # Add a helper function to fetch children
    def get_children_from_weaviate(parent_hs_code, country_code, hs_collection):
        children_rows = []
        try:
            # 1. Get the indent of the parent HTS code
            parent_obj_response = hs_collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("code").equal(parent_hs_code) &
                        weaviate.classes.query.Filter.by_property("country_code").equal(country_code),
                return_properties=["indent"],
                limit=1
            )
            if not parent_obj_response.objects:
                # Parent not found or no indent for parent, cannot determine children by indent
                print(f"    -> Warning: Parent HS code {parent_hs_code} not found or no indent for this country {country_code}.")
                return []

            parent_indent = int(parent_obj_response.objects[0].properties.get("indent", 0)) # Default to 0 if indent is missing

            # 2. Query for potential children based on code prefix, country_code, and indent level
            # HTS codes are hierarchical, so children codes usually start with the parent's code.
            response = hs_collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("code").like(f"{parent_hs_code}*") &
                        weaviate.classes.query.Filter.by_property("country_code").equal(country_code) &
                        weaviate.classes.query.Filter.by_property("indent").equal(str(parent_indent + 1)), # Filter for direct children indent
                return_properties=["country_code", "code", "description", "indent"],
            )
            for obj in response.objects:
                props = obj.properties
                child_code = props.get("code")
                # Ensure it's a child (longer code) and not the parent itself
                if child_code and len(child_code) > len(parent_hs_code):
                    children_rows.append({
                        "id": str(obj.uuid),
                        "country_code": props.get("country_code"),
                        "code": child_code,
                        "description": props.get("description"),
                        "indent": props.get("indent"),
                        "rank": -1, # Assign a very low rank for inferred children
                        "is_inferred_child": True # Flag to indicate it's an inferred child
                    })
        except Exception as e:
            print(f"    -> Error fetching children for parent {parent_hs_code}: {e}")
        return children_rows


    # 1. Fetch countries from the Logistx API
    api_url = "https://bapi.logistx.us/his_countries"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        countries_data = response.json()
        countries_list = countries_data.get("countries", [])
        country_map = {}
        for c in countries_list:
            name = str(c.get('country_name', '')).strip().upper()
            code = str(c.get('country_code', '')).strip()
            if name and code:
                country_map[name] = code
                base_name = name.split("-")[0].strip()
                if base_name: country_map[base_name] = code
        print(f"Successfully loaded {len(country_map)} countries from the API.")
    except Exception as e:
        print(f"Error fetching countries from API: {e}")
        country_map = {}

    # 2. Extract country and product descriptions from JSON, then query Weaviate
    if 'parsed_json' in globals() and 'invoices' in parsed_json:
        invoice = parsed_json['invoices'][0]
        order_level = invoice.get('order_level', {})
        receiver_country = str(order_level.get('receiver_country', '')).strip().upper()
        dest_country_code = country_map.get(receiver_country, receiver_country)
        shipper_country = str(order_level.get('shipper_country', '')).strip().upper()

        items = invoice.get('item_level', [])
        db_matches_output = []

        STOP_WORDS = {'THE', 'AND', 'FOR', 'WITH', 'USED', 'THEIR', 'FROM', 'THIS', 'THAT', 'ITS', 'INDUSTRY'}

        # Get the Weaviate collection for Hs_category
        hs_category_collection = weaviate_client.collections.get("Hs_category")

        for i, item in enumerate(items, 1):
            desc = item.get('product_description', '')
            if not desc: continue

            item_origin = str(item.get('country_of_manufacture', '')).strip().upper()
            origin_country = item_origin if item_origin else shipper_country
            orig_country_code = country_map.get(origin_country, origin_country) if origin_country else ""

            item_matches = {"sku_code": item.get("sku_code"), "product_description": desc, "matches": {}}

            words = [w.upper() for w in re.findall(r'[a-zA-Z]+', desc) if len(w) > 2]
            filtered_words = [w for w in words if w not in STOP_WORDS]

            if not filtered_words:
                db_matches_output.append(item_matches)
                continue

            search_query_text = ' '.join(filtered_words) # Combined words for Weaviate search

            codes_to_search = []
            if dest_country_code: codes_to_search.append(('Destination', dest_country_code))
            if orig_country_code and orig_country_code != dest_country_code:
                codes_to_search.append(('Origin', orig_country_code))

            for c_type, c_code in codes_to_search:
                try:
                    # Weaviate query using bm25 (keyword search)
                    response = hs_category_collection.query.bm25(
                        query=search_query_text,
                        filters=weaviate.classes.query.Filter.by_property("country_code").equal(c_code),
                        return_properties=["country_code", "code", "description", "indent"],
                        return_metadata=weaviate.classes.query.MetadataQuery(score=True)
                    )

                    rows_weaviate = []
                    seen_codes = set() # To prevent adding duplicate codes

                    # First, add direct bm25 matches
                    for obj in response.objects:
                        props = obj.properties
                        code = props.get("code")
                        if code and code not in seen_codes:
                            rank = obj.metadata.score if obj.metadata.score is not None else 0.0
                            rows_weaviate.append({
                                "id": str(obj.uuid),
                                "country_code": props.get("country_code"),
                                "code": code,
                                "description": props.get("description"),
                                "indent": props.get("indent"),
                                "rank": rank,
                                "is_inferred_child": False
                            })
                            seen_codes.add(code)

                    # Then, for each direct match, fetch its children
                    # Create a copy of codes from direct matches to iterate, as we are adding to rows_weaviate
                    direct_matches_codes = [r["code"] for r in rows_weaviate if not r["is_inferred_child"]] # Only process direct matches for children
                    for parent_code in direct_matches_codes:
                        children = get_children_from_weaviate(parent_code, c_code, hs_category_collection)
                        for child in children:
                            code = child["code"]
                            if code and code not in seen_codes:
                                rows_weaviate.append(child)
                                seen_codes.add(code)

                    # Sort by rank descending. Inferred children with rank -1 will appear at the bottom.
                    # Then by id ascending for tie-breaking.
                    rows_weaviate.sort(key=lambda x: (x['rank'], x['id']), reverse=True)

                    if rows_weaviate:
                        data = rows_weaviate
                        tree, stack = [], {}
                        for match in data:
                            lvl = int(match.get('indent', 1) or 1)
                            node = {**match, 'children': []}
                            if lvl <= 1:
                                tree.append(node)
                            else:
                                p_lvl = lvl - 1
                                while p_lvl > 0 and p_lvl not in stack: p_lvl -= 1
                                if p_lvl > 0: stack[p_lvl]['children'].append(node)
                                else: tree.append(node)
                            stack[lvl] = node
                        item_matches['matches'][c_type] = tree
                    else:
                        item_matches['matches'][c_type] = []
                except Exception as e:
                    print(f"    -> Weaviate Query failed: {e}")

            db_matches_output.append(item_matches)

        display(JSON(db_matches_output))
        os.makedirs('/content/output', exist_ok=True)
        with open('/content/output/db_matches.json', 'w', encoding='utf-8') as f:
            json.dump(db_matches_output, f, indent=2, default=str)
    else:
        print("parsed_json is missing.")
        
def update_invoice():

    if 'Qmodel' not in globals() or 'Qtokenizer' not in globals():
        print("Warning: Qwen model is not in memory. Please re-run the STEP 4 cell to load it.")
    else:
        print("Using Qwen to analyze DB matches and assign HTS codes...\n")

        db_matches_map = {}
        if 'db_matches_output' in globals():
            for match in db_matches_output:
                sku = match.get('sku_code')
                if sku:
                    db_matches_map[sku] = match.get('matches', {})
        else:
            print("Warning: db_matches_output not found.")

        invoice = parsed_json['invoices'][0]
        items = invoice.get('item_level', [])

        for i, item in enumerate(items, 1):
            sku = item.get('sku_code')
            desc = item.get('product_description')
            matches = db_matches_map.get(sku, {})

            dest_matches = matches.get('Destination', [])
            orig_matches = matches.get('Origin', [])

            # Helper to flatten the tree for the prompt
            def flatten_tree(nodes, depth=0):
                flat = []
                for n in nodes:
                    # Include indent level for better context
                    flat.append(f"- {'  ' * depth}{n['code']}: {n['description']} (Indent: {n['indent']}, Rank: {n.get('rank', 0)})")
                    if 'children' in n and n['children']:
                        flat.extend(flatten_tree(n['children'], depth + 1))
                return flat

            dest_cands = "\n".join(flatten_tree(dest_matches)[:30]) if dest_matches else "None"
            orig_cands = "\n".join(flatten_tree(orig_matches)[:30]) if orig_matches else "None"

            prompt = f"""Analyze the product and its description. Then, carefully review the provided HTS code candidates.
    Your goal is to select the most specific and accurate HTS code for both the manufacturing and destination countries based on the product's characteristics.

    If a highly specific candidate directly matches the product, choose that.
    If no specific candidate provides a perfect match, but a broader, parent-level code or an 'Other' category within a relevant parent is a better fit, select that broader code from the candidates.
    Only return 'null' if absolutely no suitable HTS code can be identified from the provided candidates, even at a broader level.

    Product Description: {desc}
    SKU: {sku}

    Origin HTS Code Candidates (Israel - Country Code: {orig_country_code}):
    {orig_cands}

    Destination HTS Code Candidates (Australia - Country Code: {dest_country_code}):
    {dest_cands}

    Return JSON ONLY:
    {{
    "manufacturing_country_hts_code": "selected_origin_hts_code_or_null",
    "destination_country_hts_code": "selected_destination_hts_code_or_null",
    "classification_reason": "Provide a concise explanation for your choice, referencing the product details and the selected HTS code's description. If you chose a broader or 'Other' code, explain why more specific options were unsuitable."
    }}
    """

            try:
                raw_response = qwen_generate([
                    {"role": "system", "content": "You are a customs assistant. Output JSON ONLY."},
                    {"role": "user", "content": prompt}
                ], max_new_tokens=500)

                try:
                    result = parse_json_maybe(raw_response)
                    item.update(result)
                    # Add the full list of HTS code candidates to the item
                    item['hts_code_candidates'] = matches
                    print(f"[{i}/{len(items)}] SKU: {sku} -> Dest: {item['destination_country_hts_code']}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for SKU {sku}: {e}")
                    print(f"Raw response from Qwen for SKU {sku}:\n{raw_response}")
                    item['classification_error'] = f"JSON parsing failed: {e}"
                    item['raw_qwen_response'] = raw_response
                    item['hts_code_candidates'] = matches # Still add candidates even if parsing fails

            except Exception as e:
                print(f"Error on SKU {sku} during Qwen generation: {e}")
                item['classification_error'] = f"Qwen generation failed: {e}"
                item['hts_code_candidates'] = matches # Still add candidates even if generation fails

        display(JSON(parsed_json))
        with open('/content/output/extracted_invoice_with_hts.json', 'w') as f:
            json.dump(parsed_json, f, indent=2)        

        
    