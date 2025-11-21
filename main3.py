def predict_urgency_from_github(
    text,
    model_dir="./result/tf_bert_urgency_model",
    threshold=0.5,
    scaler_method="power",
    use_raw_threshold=False,
):
    import os
    import warnings
    import logging
    import numpy as np

    # import tensorflow as tf
    # from transformers import AutoTokenizer
    # from keras.layers import TFSMLayer

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    warnings.filterwarnings(
        "ignore", message=".*TensorFlow and JAX classes are deprecated.*"
    )
    import tensorflow as tf
    from transformers import AutoTokenizer
    from keras.layers import TFSMLayer

    try:
        import scale_predictions
    except Exception:
        scale_predictions = None

    warnings.filterwarnings("ignore")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)

    if not os.path.exists(model_dir):
        raise RuntimeError(f"Model directory not found at {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )
    input_ids = tf.cast(encoding["input_ids"], tf.int32)
    attention_mask = tf.cast(encoding["attention_mask"], tf.int32)

    def extract_scalar_from_output(out):
        if isinstance(out, dict):
            first = list(out.values())[0]
            if isinstance(first, tf.Tensor):
                return float(first.numpy().flatten()[0])
            else:
                return float(np.asarray(first).flatten()[0])
        elif isinstance(out, (tf.Tensor,)):
            return float(out.numpy().flatten()[0])
        else:
            arr = np.asarray(out)
            return float(arr.flatten()[0])

    def apply_scaler(raw_prob):
        """
        Apply scaler if available. Returns calibrated probability and scaler name used (or None).
        """
        if scale_predictions is None:
            return raw_prob, None

        try:
            calibrated = scale_predictions.apply_scaling(raw_prob, method=scaler_method)
            try:
                calibrated_f = float(np.asarray(calibrated).flatten()[0])
            except Exception:
                calibrated_f = float(calibrated)
            return calibrated_f, scaler_method
        except Exception:
            try:
                calibrated = scale_predictions.calibrate_power(raw_prob)
                try:
                    calibrated_f = float(np.asarray(calibrated).flatten()[0])
                except Exception:
                    calibrated_f = float(calibrated)
                return calibrated_f, "power"
            except Exception:
                return raw_prob, None

    try:
        logger.info("Trying tf.keras.models.load_model(...)")
        keras_model = tf.keras.models.load_model(model_dir)
        preds = keras_model.predict((input_ids, attention_mask))
        prob_raw = float(np.asarray(preds).flatten()[0])

        prob_calibrated, scaler_used = apply_scaler(prob_raw)

        return {"signal": "urgency_keywords_present", "value": prob_calibrated}
    except Exception as e:
        logger.debug("keras load_model failed", exc_info=True)
        logger.info(
            "keras load_model not usable, falling back to SavedModel signature..."
        )

    try:
        logger.info(
            "Trying tf.saved_model.load(...) and calling serving_default signature"
        )
        loaded = tf.saved_model.load(model_dir)
        sigs = list(getattr(loaded, "signatures", {}).keys())
        if "serving_default" in sigs:
            serving_fn = loaded.signatures["serving_default"]
        else:
            serving_fn = None
            if len(sigs) > 0:
                serving_fn = loaded.signatures[sigs[0]]

        if serving_fn is None:
            raise RuntimeError("No callable signature found in SavedModel.signatures")

        try:
            _, in_sig = serving_fn.structured_input_signature
            input_names = list(in_sig.keys())
        except Exception:
            input_names = ["input_ids", "attention_mask"]

        mapping = {}
        if len(input_names) >= 2:
            mapping[input_names[0]] = input_ids
            mapping[input_names[1]] = attention_mask
        elif len(input_names) == 1:
            mapping[input_names[0]] = input_ids
        else:
            mapping = {"input_ids": input_ids, "attention_mask": attention_mask}

        out = serving_fn(**mapping)
        prob_raw = extract_scalar_from_output(out)

        prob_calibrated, scaler_used = apply_scaler(prob_raw)

        return {"signal": "urgency_keywords_present", "value": prob_calibrated}
    except Exception as e:
        logger.debug("SavedModel signature call failed", exc_info=True)
        logger.info(
            "SavedModel signature path failed, falling back to TFSMLayer if available..."
        )

    try:
        logger.info("Trying keras.layers.TFSMLayer(...)")
        from keras.layers import TFSMLayer

        tf_layer = TFSMLayer(model_dir, call_endpoint="serving_default")

        try:
            kwargs = {}
            try:
                loaded = tf.saved_model.load(model_dir)
                sigs = list(getattr(loaded, "signatures", {}).keys())
                serving_fn = (
                    loaded.signatures.get("serving_default", None) if sigs else None
                )
                if serving_fn is not None:
                    _, in_sig = serving_fn.structured_input_signature
                    input_names = list(in_sig.keys())
                else:
                    input_names = ["input_ids", "attention_mask"]
            except Exception:
                input_names = ["input_ids", "attention_mask"]

            if len(input_names) >= 2:
                kwargs[input_names[0]] = input_ids
                kwargs[input_names[1]] = attention_mask
            else:
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

            out = tf_layer(**kwargs)
        except TypeError:
            out = tf_layer((input_ids, attention_mask))

        prob_raw = extract_scalar_from_output(out)

        prob_calibrated, scaler_used = apply_scaler(prob_raw)

        return {"signal": "urgency_keywords_present", "value": prob_calibrated}
    except Exception as e:
        logger.exception("All methods to load/call the model failed.")
        raise RuntimeError(
            "Failed to load or call the model. See logs for details."
        ) from e


if __name__ == "__main__":
    examples = [
        "Your account security overview for this month shows no issues. No action needed.",
        "URGENT: Suspicious login attempt detected. Please change your password immediately!",
        "Hello. Please scan the QR code to get your OTP.",
    ]
    for t in examples:
        try:
            res = predict_urgency_from_github(t)
            print(f"Input: {t!r}\n-> {res}\n")
        except Exception as exc:
            print(f"Error when predicting for input {t!r}: {exc}\n")
